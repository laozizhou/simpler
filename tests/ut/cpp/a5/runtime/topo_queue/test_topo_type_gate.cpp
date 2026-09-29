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
 * The claim rule of the single mixed-type queue.
 *
 * The rule is: a core claims the head only when the head task matches its own
 * type, and otherwise leaves the head where it is. Each half has its own
 * failure mode. Drop the gate and an AIC kernel runs on a vector core. Turn
 * "leave it" into "skip it" -- consume the index without running the task --
 * and the task never retires, so every consumer of it spins out the timeout.
 * The cases here are written so that either mutation fails loudly.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "topo_queue_types.h"
#include "topo_worker.h"

using namespace simpler::topo_queue;

namespace {

// Single-threaded scripted policy for the deterministic step-level cases.
struct StepPolicy {
    uint32_t spins = 0;
    bool refuse_claim = false;  // models losing the CAS race to a same-type peer

    uint64_t load_queue_head(const QueueHead *head) { return head->next; }
    bool try_claim_queue_head(QueueHead *head, uint64_t expected) {
        if (refuse_claim || head->next != expected) return false;
        head->next = expected + 1;
        return true;
    }
    uint64_t load_counter_acquire(const TaskCounter *c) { return c->state; }
    void store_counter_release(TaskCounter *c, uint64_t state) { c->state = state; }
    uint64_t load_error_relaxed(const RunControl *rc) { return rc->error; }
    void latch_error(RunControl *rc, uint64_t code) {
        if (rc->error == TOPO_OK) rc->error = code;
    }
    uint64_t now_ticks() { return 0; }
    uint64_t timeout_ticks() { return 1'000; }
    void spin_hint() { ++spins; }
};

// Two independent tasks, index 0 typed AIC and index 1 typed AIV.
struct MixedPair {
    uint32_t order[2] = {0, 1};
    uint32_t fanin_offsets[3] = {0, 0, 0};  // no dependencies
    uint32_t fanin_ids[1] = {0};
    TaskCounter counters[2] = {};
    TaskEntry entries[2] = {};
    QueueHead head{};
    RunControl control{};

    MixedPair() {
        entries[0].task_id = 0;
        entries[0].core_type = CORE_TYPE_AIC;
        entries[1].task_id = 1;
        entries[1].core_type = CORE_TYPE_AIV;
    }
};

auto never_runs = [](const TaskEntry &) {
    ADD_FAILURE() << "execute must not be called on this path";
    return true;
};

}  // namespace

TEST(TopoTypeGate, AForeignHeadIsLeftUnconsumedAndTheCoreIdles)
{
    // An AIV core looking at an AIC head: no claim, no execution, the head does
    // not move, and the pass idles through spin_hint. "The head does not move"
    // is the half that kills the skip mutation -- a skipped index would show up
    // here as head.next == 1 with counters[0] still PENDING.
    MixedPair g;
    StepPolicy plat;

    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIV, g.counters, g.fanin_offsets, g.fanin_ids, g.entries,
        never_runs
    );

    EXPECT_EQ(step, WorkerStep::FOREIGN_HEAD);
    EXPECT_EQ(g.head.next, 0u) << "a foreign head must not be consumed";
    EXPECT_EQ(g.counters[0].state, TASK_PENDING);
    EXPECT_EQ(plat.spins, 1u) << "the pass must idle, not busy-loop for free";
}

TEST(TopoTypeGate, AMatchingHeadIsClaimedAndRun)
{
    MixedPair g;
    StepPolicy plat;

    bool ran = false;
    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries,
        [&](const TaskEntry &e) {
            EXPECT_EQ(e.core_type, CORE_TYPE_AIC) << "a task must only run on its own core type";
            ran = true;
            return true;
        }
    );

    EXPECT_EQ(step, WorkerStep::RAN_TASK);
    EXPECT_TRUE(ran);
    EXPECT_EQ(g.head.next, 1u);
    EXPECT_EQ(g.counters[0].state, TASK_DONE);
}

TEST(TopoTypeGate, ALostClaimHasNoSideEffectsAndRetriesImmediately)
{
    // The CAS loser must not poison anything, must not idle (the head has
    // already moved -- the next peek may be its own task), and must leave the
    // queue exactly as the winner left it.
    MixedPair g;
    StepPolicy plat;
    plat.refuse_claim = true;

    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries,
        never_runs
    );

    EXPECT_EQ(step, WorkerStep::LOST_CLAIM);
    EXPECT_EQ(g.counters[0].state, TASK_PENDING);
    EXPECT_EQ(g.counters[1].state, TASK_PENDING);
    EXPECT_EQ(g.control.error, TOPO_OK);
    EXPECT_EQ(plat.spins, 0u) << "a lost claim retries at once; idling belongs to the foreign-head path";
}

TEST(TopoTypeGate, AForeignIdlerExitsOnDrainAndOnError)
{
    MixedPair g;
    StepPolicy plat;

    // Drain: the head has moved past the end; an idler of either type leaves.
    g.head.next = 2;
    EXPECT_EQ(
        worker_step(
            plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIV, g.counters, g.fanin_offsets, g.fanin_ids, g.entries,
            never_runs
        ),
        WorkerStep::QUEUE_DRAINED
    );

    // Error latch: the only other exit an idler has, and it must fire before
    // the queue is even looked at.
    g.head.next = 0;
    g.control.error = TOPO_ERR_TASK_FAILED;
    EXPECT_EQ(
        worker_step(
            plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIV, g.counters, g.fanin_offsets, g.fanin_ids, g.entries,
            never_runs
        ),
        WorkerStep::ABORTED
    );
    EXPECT_EQ(g.head.next, 0u);
}

TEST(TopoTypeGate, AMixedGraphRunsEveryTaskExactlyOnceOnItsOwnCoreType)
{
    // The end-to-end property the gate exists for, under real concurrency:
    // AIC-typed and AIV-typed worker threads share one queue whose types
    // alternate and interleave, and every task must run exactly once, on a
    // worker of its own type, never before a predecessor finished.
    //
    // A chain of dependencies alternating across types is deliberately
    // included: an AIV task depending on an AIC task forces cross-type
    // hand-off through the head, which is where head-of-line idling and the
    // claim race actually happen.
    constexpr uint32_t kTasks = 96;
    constexpr uint32_t kWorkersPerType = 4;

    for (int attempt = 0; attempt < 30; ++attempt) {
        std::vector<uint32_t> order(kTasks);
        std::vector<uint32_t> fanin_offsets(kTasks + 1, 0);
        std::vector<uint32_t> fanin_ids;
        std::vector<TaskCounter> counters(kTasks);
        std::vector<TaskEntry> entries(kTasks);

        // Tasks in blocks of 3: AIC, AIC, AIV, AIC, AIC, AIV, ... Every task
        // past the first depends on task_id - 1, so the chain hops types
        // constantly. ids are already topological.
        for (uint32_t t = 0; t < kTasks; ++t) {
            order[t] = t;
            entries[t].task_id = t;
            entries[t].core_type = (t % 3 == 2) ? CORE_TYPE_AIV : CORE_TYPE_AIC;
            fanin_offsets[t] = static_cast<uint32_t>(fanin_ids.size());
            if (t > 0) fanin_ids.push_back(t - 1);
            fanin_offsets[t + 1] = static_cast<uint32_t>(fanin_ids.size());
        }

        QueueHead head{};
        RunControl control{};

        struct AtomicPlatform {
            uint64_t load_queue_head(const QueueHead *h) {
                return reinterpret_cast<const std::atomic<uint64_t> *>(const_cast<const uint64_t *>(&h->next))
                    ->load(std::memory_order_acquire);
            }
            bool try_claim_queue_head(QueueHead *h, uint64_t expected) {
                return reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(&h->next))
                    ->compare_exchange_strong(expected, expected + 1, std::memory_order_acq_rel);
            }
            uint64_t load_counter_acquire(const TaskCounter *c) {
                return reinterpret_cast<const std::atomic<uint64_t> *>(const_cast<const uint64_t *>(&c->state))
                    ->load(std::memory_order_acquire);
            }
            void store_counter_release(TaskCounter *c, uint64_t s) {
                reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(&c->state))
                    ->store(s, std::memory_order_release);
            }
            uint64_t load_error_relaxed(const RunControl *rc) {
                return reinterpret_cast<const std::atomic<uint64_t> *>(const_cast<const uint64_t *>(&rc->error))
                    ->load(std::memory_order_relaxed);
            }
            void latch_error(RunControl *rc, uint64_t code) {
                uint64_t expected = TOPO_OK;
                reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(&rc->error))
                    ->compare_exchange_strong(expected, code, std::memory_order_acq_rel);
            }
            uint64_t now_ticks() { return 0; }
            uint64_t timeout_ticks() { return ~0ull; }
            void spin_hint() { std::this_thread::yield(); }
        };

        // executed_on[t] records which core type actually ran task t.
        std::vector<std::atomic<int>> executed_on(kTasks);
        for (auto &e : executed_on) e.store(-1, std::memory_order_relaxed);
        std::atomic<uint32_t> total{0};

        auto spawn = [&](uint8_t type) {
            return std::thread([&, type] {
                AtomicPlatform plat;
                total.fetch_add(
                    worker_loop(
                        plat, &head, &control, order.data(), kTasks, type, counters.data(), fanin_offsets.data(),
                        fanin_ids.data(), entries.data(),
                        [&](const TaskEntry &e) {
                            const int prev = executed_on[e.task_id].exchange(type, std::memory_order_acq_rel);
                            EXPECT_EQ(prev, -1) << "task " << e.task_id << " ran twice";
                            return true;
                        }
                    ),
                    std::memory_order_relaxed
                );
            });
        };

        std::vector<std::thread> threads;
        for (uint32_t w = 0; w < kWorkersPerType; ++w) threads.push_back(spawn(CORE_TYPE_AIC));
        for (uint32_t w = 0; w < kWorkersPerType; ++w) threads.push_back(spawn(CORE_TYPE_AIV));
        for (auto &t : threads) t.join();

        ASSERT_EQ(control.error, TOPO_OK) << "attempt " << attempt;
        ASSERT_EQ(total.load(), kTasks);
        for (uint32_t t = 0; t < kTasks; ++t) {
            ASSERT_EQ(executed_on[t].load(), static_cast<int>(entries[t].core_type))
                << "task " << t << " ran on the wrong core type (attempt " << attempt << ")";
            ASSERT_EQ(counters[t].state, TASK_DONE);
        }
    }
}
