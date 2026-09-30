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
 * topo_queue AICPU executor.
 *
 * The AICPU's whole life in this runtime:
 *
 *   read HBG's uploaded graph -> build the image (topo_prepare.h) ->
 *   handshake with the cores -> publish the image -> open register windows ->
 *   wait until retired == task_count -> ring the exit bell -> clean up
 *
 * After "open register windows" it makes no scheduling decision of any kind;
 * the cores pull work themselves. The pure build logic lives in topo_prepare.h
 * so host unit tests cover it; this file owns only what needs the device:
 * registers, cache maintenance, the handshake, and the wait.
 *
 * Threading: the sim launcher starts several AICPU threads and each calls
 * aicpu_execute. topo_queue's prep is one pass over the graph -- not worth
 * partitioning in v1 -- so the first thread in elects itself leader and does
 * everything; the rest return at once. (The launcher's completion tracking
 * counts core threads too, and the cores outlive every follower by
 * construction: they exit only on the leader's bell.)
 *
 * Failure discipline: once cores are launched they spin waiting for their
 * register window, so EVERY failure path after entry still has to run the
 * release protocol -- publish a null image and open the windows -- or the run
 * hangs instead of failing. A core that reads a null image acknowledges and
 * exits immediately (aicore_executor.cpp's null-header branch).
 */

#include <atomic>
#include <cstdint>
#include <cstring>

#include "aicore_scheduler_state.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/device_malloc.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "scheduler/scheduler_watchdog.h"
#include "spin_hint.h"
#include "topo_image.h"
#include "topo_prepare.h"
#include "topo_queue_types.h"

namespace {

using namespace simpler::topo_queue;

// Waiting on cores is bring-up/teardown, not the dispatch path, so a sleepless
// spin with a watchdog is the sanctioned shape. Ten seconds mirrors the
// device-side WAIT_TIMEOUT_SECONDS.
constexpr uint64_t HANDSHAKE_TIMEOUT_CYCLES = PLATFORM_PROF_SYS_CNT_FREQ * 10;
// The whole graph must drain within this budget.
//
// It has to stay under the OS op-execute timeout, which is 45 s by default:
// that one kills aicpu-sd outright, so a budget above it means the run is
// always reaped before this one can report anything and every stall looks
// identical from the host. The margin also covers the AICore-side
// WAIT_TIMEOUT_SECONDS, which latches an error this loop reports instead.
//
// The budget does scale with the workload, unlike the handshake, so a graph
// whose honest runtime approaches this needs a per-run knob rather than a
// larger constant -- raising it past 45 s only removes the diagnostic.
constexpr uint64_t COMPLETION_TIMEOUT_CYCLES = PLATFORM_PROF_SYS_CNT_FREQ * 30;

// Leader election. The launcher serializes runs, so "owner" is per-run state:
// the first thread to swap nullptr -> runtime leads; everyone else follows and
// returns. The leader restores nullptr on its way out.
std::atomic<Runtime *> g_run_owner{nullptr};

// Full a5: 36 AIC + 72 AIV. Matches AicoreLifecycle::kMaxWorkers; a local copy
// because that one is class-scoped to HBG's lifecycle.
constexpr int32_t TOPO_MAX_CORES = 108;

struct CoreEndpoint {
    Handshake *handshake;
    uint64_t reg_addr;  // 0 = never reported; nothing to release or signal
};

/*
 * The image's GM allocation, and the two pointers a caller needs to hold.
 *
 * `header` is where the image is built and what the cores are told; `base` is
 * what the free takes. They differ because the platform allocator promises no
 * particular alignment while every wire type in the image declares a whole
 * cache line -- a counter that shared a line with its neighbour would lose a
 * completion to the other core's writeback. The base is therefore rounded up
 * inside an over-allocation, which leaves the original pointer as the only one
 * the allocator will accept back.
 */
struct TopoImageAllocation {
    TopoImageHeader *header;  // nullptr when the allocation failed
    void *base;
};

/*
 * Reserve and zero the image.
 *
 * The memory comes from the platform's device allocator, not the AICPU heap:
 * onboard, a heap pointer names AICPU-local memory that no AICore can read,
 * while this returns a device virtual address the cores reach. The two agree
 * only under simulation, where every address space is the same process.
 *
 * Zeroing is what establishes the mutable tail's initial state -- PENDING == 0,
 * head == 0, error == TOPO_OK -- so the builder writes only the read-only
 * region.
 */
TopoImageAllocation topo_image_allocate(uint32_t bytes) {
    const size_t rounded = (static_cast<size_t>(bytes) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    void *base = aicpu_device_malloc(rounded + CACHE_LINE_BYTES - 1);
    if (base == nullptr) return TopoImageAllocation{nullptr, nullptr};

    const uintptr_t aligned =
        (reinterpret_cast<uintptr_t>(base) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    void *memory = reinterpret_cast<void *>(aligned);
    std::memset(memory, 0, rounded);
    return TopoImageAllocation{static_cast<TopoImageHeader *>(memory), base};
}

/*
 * Collect every core's report. Single-threaded on purpose: this is a one-off
 * bring-up wait, and v1 trades the multi-thread partition HBG uses for less
 * machinery. Returns the number of cores collected, or -1 on timeout.
 */
int32_t collect_reports(Runtime *runtime, CoreEndpoint *endpoints, int32_t core_count) {
    Handshake *handshakes = runtime->dev.workers;
    uint64_t *regs = reinterpret_cast<uint64_t *>(get_platform_regs());
    const uint32_t physical_core_count = platform_get_physical_cores_count();
    const uint64_t report_epoch = get_platform_run_result_epoch();
    const uint64_t wait_start = get_sys_cnt_aicpu();

    int32_t remaining = core_count;
    while (remaining > 0) {
        for (int32_t i = 0; i < core_count; ++i) {
            if (endpoints[i].reg_addr != 0) continue;
            Handshake *handshake = &handshakes[i];
            cache_invalidate_range(handshake, sizeof(*handshake));
            if (!aicore_report_accepted(handshake, report_epoch)) continue;
            // The accepted marker is not an acquire; the payload loads below
            // need a real load-load barrier after it. Once per report.
            rmb();
            const uint32_t physical_core_id = handshake->physical_core_id;
            if (physical_core_id >= physical_core_count) {
                LOG_ERROR("topo_queue: core %d reported invalid physical id %u", i, physical_core_id);
                return -1;
            }
            endpoints[i].handshake = handshake;
            endpoints[i].reg_addr = regs[physical_core_id];
            --remaining;
        }
        if (remaining > 0) {
            if (scheduler_watchdog_expired(wait_start, get_sys_cnt_aicpu(), HANDSHAKE_TIMEOUT_CYCLES)) {
                LOG_ERROR("topo_queue: handshake timeout, %d cores never reported", remaining);
                return -1;
            }
            SPIN_WAIT_HINT();
        }
    }
    return core_count;
}

/*
 * Publish the image (or, with nullptr, the order to exit at once) and open
 * every reported core's register window. The task word must be in memory
 * before the window opens: the core's phase 2 reads the window as "everything
 * the AICPU published is visible", so the flush and wmb sit between the two.
 */
void release_cores(const CoreEndpoint *endpoints, int32_t core_count, TopoImageHeader *image) {
    for (int32_t i = 0; i < core_count; ++i) {
        if (endpoints[i].reg_addr == 0) continue;
        endpoints[i].handshake->task = reinterpret_cast<uint64_t>(image);
        cache_flush_range(endpoints[i].handshake, sizeof(Handshake));
    }
    wmb();
    for (int32_t i = 0; i < core_count; ++i) {
        if (endpoints[i].reg_addr == 0) continue;
        platform_init_aicore_regs(endpoints[i].reg_addr);
    }
}

void signal_exit_and_teardown(const CoreEndpoint *endpoints, int32_t core_count) {
    for (int32_t i = 0; i < core_count; ++i) {
        if (endpoints[i].reg_addr == 0) continue;
        write_reg(endpoints[i].reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    }
    for (int32_t i = 0; i < core_count; ++i) {
        if (endpoints[i].reg_addr == 0) continue;
        platform_deinit_aicore_regs(endpoints[i].reg_addr);
    }
}

/*
 * Wait for the run to finish: every task retired, or the first error latched.
 * Both live in the image's RunControl; `retired` is summed once per core as it
 * drains, so this poll converges without per-task traffic.
 */
int32_t wait_completion(TopoImageHeader *image) {
    TopoImageView view{};
    topo_image_bind(image, view);
    RunControl *control = const_cast<RunControl *>(view.run_control);
    const uint64_t task_count = image->task_count;
    const uint64_t wait_start = get_sys_cnt_aicpu();

    while (true) {
        cache_invalidate_range(control, sizeof(RunControl));
        const uint64_t error = control->error;
        if (error != TOPO_OK) {
            LOG_ERROR("topo_queue: run failed, latched error=%u", static_cast<uint32_t>(error));
            return -1;
        }
        if (control->retired == task_count) return 0;
        if (scheduler_watchdog_expired(wait_start, get_sys_cnt_aicpu(), COMPLETION_TIMEOUT_CYCLES)) {
            // The head separates a queue nobody finished draining from one that
            // drained while a claimed task never retired: head == task_count
            // means every index was claimed, so the missing retirements are
            // owned by cores that entered a task and did not leave it.
            QueueHead *head = const_cast<QueueHead *>(view.queue_head);
            cache_invalidate_range(head, sizeof(QueueHead));
            LOG_ERROR(
                "topo_queue: completion timeout, retired=%u of %u, head=%u",
                static_cast<uint32_t>(control->retired), static_cast<uint32_t>(task_count),
                static_cast<uint32_t>(head->next)
            );
            return -1;
        }
        SPIN_WAIT_HINT();
    }
}

int32_t leader_execute(Runtime *runtime) {
    const int32_t core_count = runtime->dev.worker_count;
    if (core_count <= 0 || core_count > TOPO_MAX_CORES) {
        LOG_ERROR("topo_queue: invalid worker count %d", core_count);
        return -1;
    }

    // The graph's location travels in the resident bootstrap contexts HBG's
    // host publishes. topo_queue rides that publication: it requires resident
    // mode, which also inherits the host's shape gate -- a graph with MIX,
    // SPMD or sync-start tasks selects legacy before this code ever runs, and
    // legacy is HBG's to execute, not ours.
    SchedulerWorkerContext *bootstrap = aicore_scheduler_bootstrap_context(runtime);
    if (bootstrap == nullptr || !aicore_scheduler_runtime_enabled(runtime)) {
        LOG_ERROR("%s", "topo_queue: graph is not in resident mode (unsupported shapes?); run it under host_build_graph");
        return -1;
    }
    cache_invalidate_range(bootstrap, sizeof(*bootstrap));

    const SchedulerGraphView graph{
        bootstrap->graph_storage_address,
        bootstrap->graph_reserved_address,
        bootstrap->graph_task_count,
        bootstrap->task_window_last_index,
    };
    // One pass over the read-only graph follows; make the host's upload visible.
    cache_invalidate_range(
        reinterpret_cast<void *>(graph.storage_address), graph.task_count * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE
    );

    // Cores are already spinning for their windows: from here on every exit
    // runs the release protocol, with a real image or a null one.
    CoreEndpoint endpoints[TOPO_MAX_CORES] = {};
    if (collect_reports(runtime, endpoints, core_count) < 0) {
        release_cores(endpoints, core_count, nullptr);
        signal_exit_and_teardown(endpoints, core_count);
        return -1;
    }

    TopoImageHeader sizing{};
    const uint32_t image_bytes = topo_prepare_image_bytes(graph, sizing);
    const TopoImageAllocation allocation =
        image_bytes != 0 ? topo_image_allocate(image_bytes) : TopoImageAllocation{nullptr, nullptr};
    TopoImageHeader *image = allocation.header;
    int64_t failed_task = -1;
    TopoPrepareResult prep = TOPO_PREPARE_BAD_GRAPH;
    const uint64_t *callable_addresses = reinterpret_cast<const uint64_t *>(bootstrap->callable_addresses_address);
    const uint64_t callable_count = bootstrap->callable_addresses_count;
    if (callable_addresses != nullptr && callable_count != 0) {
        cache_invalidate_range(
            const_cast<uint64_t *>(callable_addresses), callable_count * sizeof(uint64_t)
        );
    }
    if (image != nullptr) {
        *image = sizing;
        prep = topo_prepare_fill(graph, callable_addresses, callable_count, image, &failed_task);
    }
    if (image == nullptr || prep != TOPO_PREPARE_OK) {
        LOG_ERROR(
            "topo_queue: image build failed (bytes=%u, result=%u, task=%ld)", image_bytes,
            static_cast<uint32_t>(prep), static_cast<long>(failed_task)
        );
        release_cores(endpoints, core_count, nullptr);
        signal_exit_and_teardown(endpoints, core_count);
        aicpu_device_free(allocation.base);
        return -1;
    }

    // The image is complete; push it out of the AICPU's cache before any core
    // is told where it is.
    cache_flush_range(image, image->total_bytes);
    wmb();

    release_cores(endpoints, core_count, image);
    const int32_t rc = wait_completion(image);
    signal_exit_and_teardown(endpoints, core_count);

    if (rc == 0) {
        LOG_INFO("topo_queue: run complete, %u tasks retired", image->task_count);
    }
    aicpu_device_free(allocation.base);
    return rc;
}

}  // namespace

extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("%s", "topo_queue: null Runtime");
        return -1;
    }
    Runtime *expected = nullptr;
    if (!g_run_owner.compare_exchange_strong(expected, runtime, std::memory_order_acq_rel)) {
        return 0;  // follower thread; the leader does all the work this run
    }
    const int32_t rc = leader_execute(runtime);
    g_run_owner.store(nullptr, std::memory_order_release);
    return rc;
}
