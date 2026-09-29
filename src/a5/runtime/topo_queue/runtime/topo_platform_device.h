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
#pragma once

/*
 * The AICore-side Platform policy for topo_worker.h.
 *
 * Every method here owes the worker loop an ordering guarantee, and this file is
 * the only place those guarantees are turned into instructions. The unit tests
 * drive the same loop through a std::atomic policy, so a bug that lives here is
 * a bug no host test can reach -- which is why each method says what it owes
 * rather than only what it does.
 *
 * The primitives come from topo_gm_memory.h, which resolves the silicon/sim
 * split on the same memory model the repo has already debugged on real parts
 * while depending on nothing but the platform layer.
 *
 * All of them are 64-bit, which is why the wire types are: there is no 32-bit
 * atomic load or store on this device.
 */

#include <cstdint>

#include "topo_gm_memory.h"
#include "topo_queue_types.h"

namespace simpler::topo_queue {

/*
 * Ticks come from the AICore system counter. The budget is a fault backstop
 * only -- the ordering argument in topo_worker.h means a healthy run never
 * deadlocks -- so it is deliberately loose. It exists so a core whose producer
 * died reports a timeout instead of holding the device until the OS op-execute
 * watchdog kills the whole process, which would lose the diagnostic.
 *
 * What it must cover is wider than "waiting for one producer", and sizing it
 * as if it were is the way to get spurious failures. A worker starts its clock
 * the moment its claim CAS wins an index, which can be long before its
 * predecessors run: with W workers the in-flight window is up to W-1 tasks, and
 * the budget has to cover that whole window draining. Measured on the host
 * policy: one chain, 1 worker completes healthily, and the same chain with 4
 * workers and the same budget times out -- marking FAILED tasks that never ran.
 * At 108 cores and microsecond kernels the margin here is still ~1000x, but a
 * graph of long kernels needs this raised, and it should become a per-run knob
 * rather than a constant before this runtime carries real work.
 */
inline constexpr uint64_t WAIT_TIMEOUT_SECONDS = 10;

struct DevicePlatform {
    uint64_t timeout_cycles;

    explicit __aicore__ DevicePlatform(uint64_t sys_cnt_freq) :
        timeout_cycles(sys_cnt_freq * WAIT_TIMEOUT_SECONDS) {}

    /*
     * Peek the head index. Owes: nothing. ld_dev reads GM directly and a stale
     * value costs one failed claim or one redundant peek -- ownership is decided
     * only by the CAS below, never by this read. The line is invalidated first
     * so an idling core cannot spin forever on a copy it cached before the last
     * claim moved the head.
     */
    __aicore__ uint64_t load_queue_head(const QueueHead *head) {
        topo_observe_cache_line(const_cast<QueueHead *>(head));
        return topo_gm_query(const_cast<QueueHead *>(head)->next);
    }

    /*
     * Claim queue index `expected` by advancing the head past it. This is the
     * only contended write on the hot path: every worker targets the same line,
     * which is why QueueHead is padded to own one.
     *
     * Owes: atomicity, and acq_rel so a won claim orders after every peek that
     * justified it. Winning means exclusive ownership of index `expected`.
     */
    __aicore__ bool try_claim_queue_head(QueueHead *head, uint64_t expected) {
        return topo_gm_compare_exchange(head->next, expected, expected + 1) == expected;
    }

    /*
     * Read a peer's completion counter.
     *
     * Owes: acquire. A worker that observes TASK_DONE here goes on to read the
     * producer's output tensors, and the counter's address does not depend on
     * the output address -- so without a barrier the core is free to satisfy the
     * output load first and see pre-task bytes. This is the same hazard
     * docs/hardware/cache-coherency.md records for the AICPU side of the
     * AICore->AICPU path, and it has the same answer: a real barrier, not a
     * data dependency.
     *
     * The dcci is what makes the peer's write visible at all: ld_dev bypasses
     * the scalar DCache but the line may still be held stale, so the line is
     * invalidated before it is read.
     */
    __aicore__ uint64_t load_counter_acquire(const TaskCounter *counter) {
        topo_observe_cache_line(const_cast<TaskCounter *>(counter));
        const uint64_t state = topo_gm_query(const_cast<TaskCounter *>(counter)->state);
        if (state != TASK_PENDING) {
            // Terminal state observed: order every later load after this one,
            // so the task's outputs cannot be read from before it ran.
            topo_full_barrier();
        }
        return state;
    }

    /*
     * Publish this task's terminal state.
     *
     * Owes: release, and the flush that gives release meaning across cores. The
     * task's output tensors are ordinary GM writes sitting in this core's cache;
     * a consumer that sees DONE must be able to read them. So the sequence is
     * flush -> barrier -> store, in that order, and never the store first.
     *
     * The store itself carries OUT_OF_ORDER_STORE_BARRIER() inside
     * topo_gm_store, so the counter cannot be reordered ahead of the
     * barrier below it either.
     */
    __aicore__ void store_counter_release(TaskCounter *counter, uint64_t state) {
        // A real writeback, not just a barrier. The kernel's scalar GM stores
        // sit dirty in this core's data cache, while a consumer reads those same
        // addresses through ld_dev, which bypasses it -- so a barrier alone
        // would order a flush that never happened and the consumer would read
        // pre-task bytes behind a counter that says DONE. topo_publish_all()
        // ends with the barrier, so the store below cannot precede it.
        topo_publish_all();
        topo_gm_store(counter->state, state);
    }

    /*
     * Poll the run-wide error latch.
     *
     * Owes: nothing for correctness. A task that must not run is held by its
     * predecessor's counter, and that path invalidates; this poll only lets a
     * doomed worker leave sooner than the counter or the budget would make it.
     *
     * It still invalidates, because an accelerator that can be arbitrarily
     * stale is not one. Reading without the dcci would leave this core free to
     * satisfy the poll from a line it has held since before the error was
     * latched, for as long as it keeps spinning -- so the early exit would fire
     * only by luck, and the observable failure would be a
     * TOPO_ERR_WAIT_TIMEOUT that hides the real cause. The repo's own rule for
     * published metadata is the same (scheduler_memory.h: "publication-
     * protected metadata is invalidated separately before consumption").
     *
     * The cost is one line invalidation per spin pass on a line that is
     * read-mostly and written at most once per run, which is the wrong shape
     * for the traffic it generates across a full device. If that ever shows up
     * in a profile, the fix is to sample it -- invalidate every Nth pass rather
     * than every pass -- not to drop the invalidation, since bounded staleness
     * is the property worth keeping.
     */
    __aicore__ uint64_t load_error_relaxed(const RunControl *control) {
        topo_observe_cache_line(const_cast<RunControl *>(control));
        return topo_gm_query(const_cast<RunControl *>(control)->error);
    }

    /*
     * Latch the first error. Later writers must not overwrite it: the first
     * failure is the one that explains the run, and the ones that follow are
     * usually its consequences.
     */
    __aicore__ void latch_error(RunControl *control, uint64_t code) {
        topo_gm_compare_exchange(control->error, static_cast<uint64_t>(TOPO_OK), code);
    }

    __aicore__ uint64_t now_ticks() { return get_sys_cnt_aicore(); }

    __aicore__ uint64_t timeout_ticks() { return timeout_cycles; }

    /*
     * One iteration of the predecessor wait.
     *
     * Must not sleep: .claude/rules/codestyle.md rule 5 bans that anywhere a
     * task's latency passes through. Whether it may yield is decided per
     * variant by the platform's own SPIN_WAIT_HINT, which topo_spin_hint()
     * carries -- ((void)0) on silicon, where there is no scheduler to yield to,
     * and a yield under simulation, where this core is a host thread that would
     * otherwise starve the producer it is waiting on.
     */
    __aicore__ void spin_hint() { topo_spin_hint(); }
};

}  // namespace simpler::topo_queue
