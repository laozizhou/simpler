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
 * The worker loop every AICore runs: peek the queue head, claim it if the task
 * is this core's type, wait on predecessors, execute, publish.
 *
 * The loop is a template over a Platform policy so the same source is the one
 * that runs on silicon and the one the unit tests drive. The device policy maps
 * onto topo_gm_* and the AICore cache intrinsics; the test policy maps onto
 * std::atomic, which lets a host thread pool exercise the ordering rules with no
 * device in the picture. Anything that would differ between them is expressed as
 * a policy call rather than an #ifdef, so there is no path that only one of the
 * two ever compiles.
 *
 * ---------------------------------------------------------------------------
 * One queue over two core types: the claim rule
 * ---------------------------------------------------------------------------
 * There is a single queue, in topological order, holding AIC and AIV tasks
 * interleaved. Every a5 core enters the same executor, so a core may find the
 * head task typed for the other kind of core. The rule is:
 *
 *   a core CLAIMS the head only when the head task matches its own type;
 *   otherwise it leaves the head where it is and idles.
 *
 * The claim is a CAS (head: h -> h+1), not a blind fetch_add, and that is
 * load-bearing. With fetch_add, "skip a foreign task" hands the index to a core
 * that will never run it -- nothing puts an index back -- so the task never
 * retires and every consumer of it spins out the timeout. Peek-then-CAS means an
 * index is only ever consumed by a core that will actually execute it.
 *
 * The cost accepted here is head-of-line idling: while the head is (say) AIC and
 * every AIC core is busy, all AIV cores idle even if runnable AIV tasks sit
 * right behind the head. That is a throughput tax, not a correctness risk, and
 * it was chosen deliberately over per-type queues for the simpler protocol; the
 * per-type-queue variant exists in this repo's history if measurement ever says
 * the tax is too high.
 *
 * ---------------------------------------------------------------------------
 * Why this cannot deadlock
 * ---------------------------------------------------------------------------
 * Let S be the set of tasks that have been claimed but not yet retired, and let
 * m be the lowest queue index in S. Every index below m was claimed earlier
 * (the head only moves forward, by one, at each claim) and is not in S, so it
 * has retired. Every predecessor of m sits at a lower index (topological
 * order), hence has retired: m's holder can always run.
 *
 * The head itself can never stall forever either. Say the head task is of type
 * T. Either some T core is free -- it claims the head -- or every T core holds
 * a claimed task; the lowest-indexed claimed task overall is runnable by the
 * argument above, and if it belongs to another type its holder retires it and
 * shrinks S, so eventually the lowest one is a T task, its holder finishes,
 * returns to the head, and claims. Progress never stops.
 *
 * The argument needs three invariants: the queue is in topological order, the
 * head advances only by a successful claim (no skipping), and a claimed task is
 * always driven to a terminal counter state (every exit path below publishes
 * DONE or FAILED). Work stealing or queue reordering forfeits the property and
 * must bring its own proof.
 *
 * ---------------------------------------------------------------------------
 * Memory ordering
 * ---------------------------------------------------------------------------
 * Two orderings are load-bearing and neither is checkable by a test on sim,
 * whose cache model is a no-op:
 *
 *   publish: write the task's outputs -> flush them to GM -> barrier -> only
 *            then store DONE. A consumer that sees DONE must be able to read
 *            every byte the producer wrote.
 *
 *   consume: read DONE -> barrier -> only then read the producer's outputs.
 *            The counter address does not depend on the output address, so
 *            nothing stops the core from satisfying the output load first.
 *
 * The peek itself needs no ordering beyond the claim CAS: order[] and entries[]
 * are immutable after bring-up, and a stale head read costs one failed CAS,
 * never a wrong claim.
 *
 * These mirror the AICore->AICPU protocol in docs/hardware/cache-coherency.md,
 * for the same reason: an acquire on the flag has to be a real barrier, not an
 * implied dependency.
 *
 * ---------------------------------------------------------------------------
 * What a failed run leaves behind
 * ---------------------------------------------------------------------------
 * RunControl::error is the authoritative verdict; the counters are not. A worker
 * that observes the latched error returns instead of claiming further tasks, so
 * a consumer nobody reached stays PENDING rather than FAILED. Exiting fast is
 * the point -- walking the remaining graph only to mark it would cost the whole
 * queue -- but it means a counter answers "did this task retire", never "did the
 * run succeed". Poisoning happens only where a worker actually claimed a task
 * whose predecessor had already failed, and exists so that anything already
 * spinning on that task stops without waiting out the timeout.
 */

#include <cstdint>

#include "topo_queue_types.h"

namespace simpler::topo_queue {

/*
 * What a worker reports back for one loop iteration. The caller drives the loop,
 * so the policy stays free of control flow.
 */
enum class WorkerStep : uint8_t {
    RAN_TASK,      // claimed a task, waited, executed, published
    FOREIGN_HEAD,  // head task belongs to the other core type; idled one pass
    LOST_CLAIM,    // another core of this type claimed the head first; retry now
    QUEUE_DRAINED, // head is past the end; this worker is done
    ABORTED,       // run_control->error was set, or this worker latched it
};

/*
 * Platform policy contract. A conforming policy provides:
 *
 *   uint64_t load_queue_head(const QueueHead *)
 *       Observation read of the head index. May be stale: a stale value costs
 *       one failed claim or one extra peek, never a wrong claim -- the CAS
 *       below is what decides ownership.
 *
 *   bool try_claim_queue_head(QueueHead *, uint64_t expected)
 *       Atomic compare-exchange of the head from `expected` to `expected + 1`.
 *       True means this worker owns queue index `expected`, exclusively. This
 *       is the only contended write on the hot path.
 *
 *   uint64_t load_counter_acquire(const TaskCounter *)
 *       Read a peer's counter with acquire semantics: no later load may be
 *       satisfied before it. On device this is a GM read plus a barrier.
 *
 *   void store_counter_release(TaskCounter *, uint64_t state)
 *       Publish this task's terminal state after flushing its outputs. On device
 *       this is the cache flush, the barrier, and then the store.
 *
 *   uint64_t load_error_relaxed(const RunControl *)
 *       Cheap poll; may be stale by a bounded amount.
 *
 *   void latch_error(RunControl *, uint64_t code)
 *       First writer wins; later writers must not overwrite a latched code.
 *
 *   uint64_t now_ticks()  /  uint64_t timeout_ticks()
 *       Fault budget for the predecessor wait. Correctness never depends on it.
 *
 *   void spin_hint()
 *       One iteration of a wait. Must not sleep: .claude/rules/codestyle.md
 *       rule 5 allows a dispatch-path wait only to spin or block on a wakeup
 *       primitive. Whether it may yield belongs to the platform, not to this
 *       contract -- on silicon there is no scheduler to yield to and the hint is
 *       empty, while under simulation a core is a host thread and must yield or
 *       it starves the producer it waits on.
 */

/*
 * Run one task if the head offers one of this core's type.
 *
 * `execute` is invoked as execute(const TaskEntry &) and returns true on
 * success. It is a template parameter rather than a function pointer so the
 * device build can inline the dispatch.
 */
template <typename Platform, typename Execute>
WorkerStep worker_step(
    Platform &plat,
    QueueHead *head,
    RunControl *control,
    const uint32_t *order,        // queue: order[i] is the task id at index i
    uint32_t task_count,
    uint8_t self_core_type,       // CORE_TYPE_AIC or CORE_TYPE_AIV
    TaskCounter *counters,        // counters[task_id]
    const uint32_t *fanin_offsets,
    const uint32_t *fanin_ids,
    const TaskEntry *entries,     // entries[task_id]
    Execute &&execute
) {
    // Checked every pass, including foreign-head idle passes: the error latch
    // and the drain below are an idling core's only exits, since a foreign head
    // may legitimately hold it for as long as the other type's backlog takes.
    if (plat.load_error_relaxed(control) != TOPO_OK) return WorkerStep::ABORTED;

    const uint64_t index = plat.load_queue_head(head);
    if (index >= task_count) return WorkerStep::QUEUE_DRAINED;

    const uint32_t task_id = order[index];
    const TaskEntry &entry = entries[task_id];

    if (entry.core_type != self_core_type) {
        // Not ours. Leave the head exactly where it is -- consuming it would
        // hand the index to a core that cannot run it, and nothing puts an
        // index back. A core of the matching type will claim it; this one
        // idles a pass. Deliberately no deadline here: idling at a foreign
        // head is legitimate for as long as the other type's backlog takes,
        // and a genuinely dead run still reaches this core through the error
        // latch above (whoever times out on a predecessor latches it).
        plat.spin_hint();
        return WorkerStep::FOREIGN_HEAD;
    }

    if (!plat.try_claim_queue_head(head, index)) {
        // A same-type peer won the race; the head has moved. Re-peek at once --
        // the next index may still be ours.
        return WorkerStep::LOST_CLAIM;
    }

    // From here on this worker owns queue index `index` exclusively, and every
    // exit path must drive counters[task_id] to a terminal state -- the
    // deadlock-freedom argument counts on claimed tasks always retiring.

    // Wait for every predecessor, advancing a cursor that never goes back: a
    // counter is monotone once terminal, so a satisfied predecessor is never
    // re-read. The scan is therefore O(fanin) over the whole wait, not per
    // pass -- each pass re-reads only the one predecessor still blocking.
    // Keeping the state to a single counter per task is what lets it work with
    // nothing to register or unregister, where HBG's classify_fanin_state
    // maintains per-task dependency records instead.
    const uint32_t begin = fanin_offsets[task_id];
    const uint32_t end = fanin_offsets[task_id + 1];
    const uint64_t deadline = plat.now_ticks() + plat.timeout_ticks();

    for (uint32_t e = begin; e < end;) {
        const uint32_t producer = fanin_ids[e];
        const uint64_t state = plat.load_counter_acquire(&counters[producer]);

        if (state == TASK_DONE) {
            ++e;  // this one is satisfied; it is monotonic, so never re-check it
            continue;
        }
        if (state == TASK_FAILED) {
            // Do not run, and do not publish DONE: this task's consumers must
            // see the failure too. Publishing FAILED propagates it down the
            // graph without every waiter having to poll run_control.
            plat.latch_error(control, TOPO_ERR_PREDECESSOR_FAILED);
            plat.store_counter_release(&counters[task_id], TASK_FAILED);
            return WorkerStep::ABORTED;
        }

        // Still pending. A run-wide error means no one will ever retire it.
        if (plat.load_error_relaxed(control) != TOPO_OK) {
            plat.store_counter_release(&counters[task_id], TASK_FAILED);
            return WorkerStep::ABORTED;
        }
        if (plat.now_ticks() > deadline) {
            plat.latch_error(control, TOPO_ERR_WAIT_TIMEOUT);
            plat.store_counter_release(&counters[task_id], TASK_FAILED);
            return WorkerStep::ABORTED;
        }
        plat.spin_hint();
    }

    const bool ok = execute(entry);

    if (!ok) {
        // Latch before publishing, matching the predecessor path above. The
        // other order loses the root cause: a consumer already spinning on this
        // counter observes FAILED the moment it is stored and latches
        // TOPO_ERR_PREDECESSOR_FAILED, so whichever of the two reaches the CAS
        // first wins -- and the run would then be explained by a consequence
        // rather than by the task that actually failed.
        plat.latch_error(control, TOPO_ERR_TASK_FAILED);
        plat.store_counter_release(&counters[task_id], TASK_FAILED);
        return WorkerStep::ABORTED;
    }

    // store_counter_release owes the flush-then-barrier-then-store sequence; a
    // consumer that observes DONE must see every byte the task wrote.
    plat.store_counter_release(&counters[task_id], TASK_DONE);
    return WorkerStep::RAN_TASK;
}

/*
 * Drive worker_step until the queue drains or the run fails, and report how many
 * tasks this worker retired. The caller adds that to RunControl::retired in one
 * batch: a per-task increment would put every worker on the same contended line
 * for the whole run, which is the cost a5's resident scheduler batches away.
 */
template <typename Platform, typename Execute>
uint32_t worker_loop(
    Platform &plat,
    QueueHead *head,
    RunControl *control,
    const uint32_t *order,
    uint32_t task_count,
    uint8_t self_core_type,
    TaskCounter *counters,
    const uint32_t *fanin_offsets,
    const uint32_t *fanin_ids,
    const TaskEntry *entries,
    Execute &&execute
) {
    uint32_t retired = 0;
    for (;;) {
        const WorkerStep step = worker_step(
            plat, head, control, order, task_count, self_core_type, counters, fanin_offsets, fanin_ids, entries,
            execute
        );
        if (step == WorkerStep::RAN_TASK) {
            ++retired;
            continue;
        }
        if (step == WorkerStep::FOREIGN_HEAD || step == WorkerStep::LOST_CLAIM) {
            continue;  // the step already idled (foreign) or must re-peek now (lost)
        }
        return retired;  // QUEUE_DRAINED or ABORTED
    }
}

}  // namespace simpler::topo_queue
