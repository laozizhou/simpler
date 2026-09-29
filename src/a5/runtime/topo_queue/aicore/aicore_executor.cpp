/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*
 * topo_queue AICore executor — the self-pulling resident loop.
 *
 * The AICPU publishes one queue and then stops deciding anything. Each core
 * peeks the shared head, claims it by CAS only when the head task matches its
 * own type, waits for that task's predecessors to publish DONE, runs it, and
 * publishes its own counter. There
 * is no dispatch register write per task, no ready queue, and no per-core
 * mailbox: the register protocol is used only for bring-up and exit.
 *
 * The bring-up and exit handshake is deliberately the legacy one
 * (aicore_legacy_executor.cpp), not the resident scheduler's. Legacy talks to
 * the AICPU through `Handshake` and two registers and nothing else, so this
 * runtime borrows a protocol rather than a data model -- none of a5 HBG's
 * scheduler types cross into here, and their churn cannot reach this file.
 *
 * What the loop itself does lives in runtime/topo_worker.h, which the unit
 * tests drive through a host policy. This file supplies the device policy and
 * the shell around it.
 */

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/core_type.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "topo_image.h"
#include "topo_platform_device.h"
#include "topo_queue_types.h"
#include "topo_worker.h"

using namespace simpler::topo_queue;

// This is the one translation unit that sees both the platform's CoreType and
// the byte encoding TaskEntry carries, so it is where a divergence between them
// can still be caught at compile time.
static_assert(static_cast<int32_t>(::CoreType::AIC) == CORE_TYPE_AIC);
static_assert(static_cast<int32_t>(::CoreType::AIV) == CORE_TYPE_AIV);

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

namespace {

/*
 * Run one task's kernel.
 *
 * The kernel's own input dcci runs inside the kernel, strictly after the
 * predecessor wait the caller already completed, so the producer's outputs are
 * visible to it. The trailing barrier orders this core's stores before the
 * caller publishes the completion counter -- store_counter_release adds its own
 * flush, but the kernel's last write must be committed before that flush is
 * allowed to be reordered against it.
 */
// The writer (topo_prepare.h) pins the same equality; a stride disagreement is
// a version skew between the aicpu and aicore binaries and is caught at bind.
static_assert(sizeof(DispatchPayload) % CACHE_LINE_BYTES == 0, "payload stride must be whole cache lines");

__aicore__ __attribute__((always_inline)) bool
execute_task(const TopoImageView &image, const TaskEntry &entry) {
    // The view is a local object holding __gm__ pointers -- it is not itself in
    // GM, so it is passed by an ordinary reference. Qualifying it __gm__ would
    // be an address-space error the sim build cannot see, since __gm__ expands
    // to nothing there.
    //
    // No core-type check here: worker_step's claim gate is the single authority
    // on type matching -- a task only reaches this point on a core whose type it
    // matches.
    //
    // The payload was materialized by HBG's own routine on the AICPU, so its
    // args carry the tensor addresses, scalars and context slots exactly as the
    // unified kernel signature expects -- the same unpacking legacy does.
    __gm__ DispatchPayload *payload =
        reinterpret_cast<__gm__ DispatchPayload *>(topo_image_payload(image, entry.task_id));
    const uint64_t func_addr = payload->function_bin_addr;
    if (func_addr == 0) {
        // A task whose callable did not resolve is a bind-time error that
        // reached the device. Report it rather than silently retiring: the
        // graph's outputs would be wrong in a way no golden check attributes
        // back here.
        return false;
    }
    UnifiedKernelFunc kernel = reinterpret_cast<UnifiedKernelFunc>(func_addr);
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
    return true;
}

}  // namespace

/*
 * Entry point. Weak so a test or a later variant can override it at link time,
 * matching what a5 HBG does.
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *handshake = reinterpret_cast<__gm__ Handshake *>(&runtime->dev.workers[block_idx]);

    // Phase 1 -- report identity and signal arrival. Both fields are self-known,
    // so no wait is needed; the AICPU opens this core's register window only
    // after it observes aicore_done, which makes one report enough. The host
    // clears aicore_done before launch, so what the AICPU reads is this run's
    // report and never a stale one.
    const uint64_t report_epoch = get_aicore_report_epoch();
    handshake->physical_core_id = get_physical_core_id();
    handshake->core_type = core_type;
    if (report_epoch != 0) {
        // Native program run: aicore_done is payload and report_epoch is the
        // marker that commits it, so the barrier has to sit between them.
        handshake->aicore_done = block_idx + 1;
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->report_epoch = report_epoch;
    } else {
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->aicore_done = block_idx + 1;
    }
    dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Phase 2 -- wait for the AICPU to open the register window. A kernel launch
    // resets DATA_MAIN_BASE to 0; the AICPU writes a non-zero value as it opens
    // the window, so a non-zero read means reads and writes are valid. The
    // window opening is also the sync point for everything the AICPU published
    // before it, which is how the image address below becomes safe to read.
    while (read_reg(RegId::DATA_MAIN_BASE) == 0) {
        SPIN_WAIT_HINT();
    }
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    // The AICPU wrote the image descriptor into the handshake before opening the
    // window, so re-read the line to get its published value rather than the
    // copy this core flushed in phase 1.
    dcci(handshake, SINGLE_CACHE_LINE);
    __gm__ TopoImageHeader *header = reinterpret_cast<__gm__ TopoImageHeader *>(handshake->task);
    if (header == nullptr) {
        // No image means no RunControl, so this core cannot latch an error and
        // cannot join the Phase-5 exit rendezvous either -- it leaves at once,
        // which is the one path that acknowledges without having waited for
        // AICORE_EXIT_SIGNAL. An AICPU that opened the register window without
        // publishing an image is a bring-up bug on its side; this branch keeps
        // the core from hanging on it rather than defining a second protocol.
        OUT_OF_ORDER_STORE_BARRIER();
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
        return;
    }
    // The header is wider than one line, so invalidate it by size: a field left
    // in an uninvalidated line would be read stale, and every array below is
    // resolved through those fields.
    topo_observe_range(header, sizeof(TopoImageHeader));

    if (header->magic != TOPO_IMAGE_MAGIC || header->version != TOPO_IMAGE_VERSION) {
        // The header is garbage -- on silicon this is what "the AICPU's image
        // address is not real GM" looks like from a core. NOTHING in it can be
        // trusted, including run_control_offset, so latching an error here
        // would CAS a wild GM address and turn a diagnosable failure into
        // random corruption. Write nothing; leave through the null-image door.
        // The AICPU then reports a completion timeout with retired=0, which the
        // triage table maps straight to this cause.
        OUT_OF_ORDER_STORE_BARRIER();
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
        return;
    }

    TopoImageView image;
    topo_image_bind(header, image);

    if (image.payload_stride != sizeof(DispatchPayload)) {
        // Version skew between the aicpu .so that built the image and this
        // aicore binary. The magic/version check above passed, so the header --
        // and therefore run_control -- is trustworthy enough to latch through:
        // the AICPU reports a cause instead of a completion timeout. Then leave
        // through the same door as a null image.
        topo_gm_compare_exchange(image.run_control->error, static_cast<uint64_t>(TOPO_OK), TOPO_ERR_BAD_IMAGE);
        OUT_OF_ORDER_STORE_BARRIER();
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
        return;
    }

    // The tables the loop reads are published once by the AICPU before any core
    // starts and are then read with ordinary cached loads, so they are observed
    // once here rather than per access. Everything from the header base up to
    // the counters is that read-only region; the mutable objects past it carry
    // their own invalidation on each access.
    topo_observe_range(header, header->counters_offset);

    // One queue for every core; the type gate lives in worker_step's claim rule.
    // This core only claims head tasks matching its own type, and idles when the
    // head belongs to the other type -- never skipping, since a consumed index
    // that nobody runs would strand every consumer of that task.
    const uint8_t self_core_type = (core_type == ::CoreType::AIV) ? CORE_TYPE_AIV : CORE_TYPE_AIC;

    DevicePlatform plat(PLATFORM_PROF_SYS_CNT_FREQ);

    // Phase 3 -- the resident loop. worker_loop returns when the queue drains or
    // the run fails; both are terminal for this core.
    const uint32_t retired = worker_loop(
        plat, image.queue_head, image.run_control, image.order, image.header->task_count, self_core_type,
        image.counters, image.fanin_offsets, image.fanin_ids, image.entries,
        [&](const TaskEntry &entry) { return execute_task(image, entry); }
    );

    // Phase 4 -- publish this core's contribution in one add. Per-task increments
    // would put every core on this one line for the whole run; a single add at
    // the end is what keeps the line cold, the same trade a5's resident
    // scheduler makes with scheduler_flush_completions.
    if (retired != 0) {
        topo_gm_fetch_add(image.run_control->retired, static_cast<uint64_t>(retired));
    }

    // Phase 5 -- wait for the AICPU's exit signal, then acknowledge. The AICPU
    // needs every core's `retired` contribution before it can decide the run is
    // complete, so a core that drained early parks here instead of leaving: its
    // exit acknowledgement is what the AICPU counts.
    while (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL) {
        SPIN_WAIT_HINT();
    }

    // The AICPU reads this acknowledgement as the point this core's writes have
    // landed, so the preceding GM stores must complete first.
    OUT_OF_ORDER_STORE_BARRIER();
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
}
