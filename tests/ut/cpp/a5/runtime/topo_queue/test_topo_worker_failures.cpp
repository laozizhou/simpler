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
 * The failure branches of worker_step, one case each.
 *
 * These exist because mutation testing showed the rest of the suite does not
 * reach them: the pre-pull error check, the predecessor-FAILED branch, the
 * in-spin error poll and the wait timeout could each be deleted outright with
 * every other topo_queue case still green. Two of the five error codes had no
 * coverage at all.
 *
 * They are hard to reach through the normal path precisely because the design
 * works: a healthy run never times out, and once an error is latched the
 * pre-pull check stops everyone before they can enter the branches below. Each
 * case therefore drives the Platform policy -- the seam the loop is already
 * built on -- to put a worker in the exact state the branch is written for,
 * rather than racing to produce it.
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

/*
 * A policy with a driven clock and an optionally-hidden error latch.
 *
 * `blind_to_error` models a worker that passed the pre-pull check just before
 * another core latched the error -- the only way the in-loop branches are
 * reachable in production, and a race rather than a state a test can schedule.
 */
struct ScriptedPlatform {
    uint64_t clock = 0;
    uint64_t clock_step_per_spin = 0;
    uint64_t budget = 1'000;
    bool blind_to_error = false;
    uint32_t spins = 0;

    uint64_t load_queue_head(const QueueHead *head) { return head->next; }
    bool try_claim_queue_head(QueueHead *head, uint64_t expected) {
        if (head->next != expected) return false;
        head->next = expected + 1;
        return true;
    }
    uint64_t load_counter_acquire(const TaskCounter *c) { return c->state; }
    void store_counter_release(TaskCounter *c, uint64_t state) { c->state = state; }
    uint64_t load_error_relaxed(const RunControl *rc) { return blind_to_error ? TOPO_OK : rc->error; }
    void latch_error(RunControl *rc, uint64_t code) {
        if (rc->error == TOPO_OK) rc->error = code;
    }
    uint64_t now_ticks() { return clock; }
    uint64_t timeout_ticks() { return budget; }
    void spin_hint() {
        ++spins;
        clock += clock_step_per_spin;
    }
};

// Two tasks, 1 depends on 0, queued in that order.
struct Chain2 {
    uint32_t order[2] = {0, 1};
    uint32_t fanin_offsets[3] = {0, 0, 1};  // task 0: none; task 1: one
    uint32_t fanin_ids[1] = {0};
    TaskCounter counters[2] = {};
    TaskEntry entries[2] = {};
    QueueHead head{};
    RunControl control{};
};

auto never_runs = [](const TaskEntry &) {
    ADD_FAILURE() << "execute must not be called on this path";
    return true;
};

}  // namespace

TEST(TopoWorkerFailures, ALatchedErrorStopsAWorkerBeforeItClaimsAnIndex)
{
    // The pre-pull check. Deleting it is not merely wasteful: the worker would
    // claim a queue slot it then abandons, and nothing puts that slot back.
    Chain2 g;
    g.control.error = TOPO_ERR_TASK_FAILED;
    ScriptedPlatform plat;

    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries, never_runs
    );

    EXPECT_EQ(step, WorkerStep::ABORTED);
    EXPECT_EQ(g.head.next, 0u) << "an aborting worker must not consume a queue index";
    EXPECT_EQ(g.counters[0].state, TASK_PENDING);
    EXPECT_EQ(g.counters[1].state, TASK_PENDING);
}

TEST(TopoWorkerFailures, AFailedPredecessorPoisonsTheConsumerAndLatchesItsOwnCode)
{
    // The branch a consumer takes when it is already inside the fanin scan as
    // its producer fails. Poisoning is what stops anything waiting on THIS task
    // from spinning out the full timeout behind a producer that will never
    // retire.
    Chain2 g;
    g.counters[0].state = TASK_FAILED;
    g.head.next = 1;  // the consumer claims index 1, i.e. task 1

    ScriptedPlatform plat;
    plat.blind_to_error = true;  // got past the pre-pull check before the latch

    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries, never_runs
    );

    EXPECT_EQ(step, WorkerStep::ABORTED);
    EXPECT_EQ(g.counters[1].state, TASK_FAILED) << "the consumer must poison itself, not stay pending";
    EXPECT_EQ(g.control.error, TOPO_ERR_PREDECESSOR_FAILED);
}

TEST(TopoWorkerFailures, APendingPredecessorPlusALatchedErrorPoisonsTheHeldTask)
{
    // The in-spin poll. Without it a worker holding a task whose producer will
    // never run spins until the timeout instead of leaving at once -- and
    // leaves its own consumers waiting that much longer.
    Chain2 g;
    g.counters[0].state = TASK_PENDING;
    g.head.next = 1;

    // Blind on the pre-pull check, sighted afterwards: exactly the ordering the
    // branch exists for.
    struct OnceBlindPlatform : ScriptedPlatform {
        int calls = 0;
        uint64_t load_error_relaxed(const RunControl *rc) { return (calls++ == 0) ? TOPO_OK : rc->error; }
    } plat;
    g.control.error = TOPO_ERR_TASK_FAILED;

    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries, never_runs
    );

    EXPECT_EQ(step, WorkerStep::ABORTED);
    EXPECT_EQ(g.counters[1].state, TASK_FAILED);
    EXPECT_EQ(g.control.error, TOPO_ERR_TASK_FAILED) << "the first code must survive";
}

TEST(TopoWorkerFailures, AWaitThatOutlastsItsBudgetLatchesTimeoutAndPoisons)
{
    // The fault backstop. Correctness never depends on it, but a core that died
    // mid-task must not hold the device until the OS op-execute watchdog kills
    // the process -- that loses the diagnostic entirely.
    Chain2 g;
    g.counters[0].state = TASK_PENDING;  // a producer that never retires
    g.head.next = 1;

    ScriptedPlatform plat;
    plat.budget = 10;
    plat.clock_step_per_spin = 100;  // one spin overshoots the deadline

    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries, never_runs
    );

    EXPECT_EQ(step, WorkerStep::ABORTED);
    EXPECT_EQ(g.control.error, TOPO_ERR_WAIT_TIMEOUT);
    EXPECT_EQ(g.counters[1].state, TASK_FAILED);
    EXPECT_GT(plat.spins, 0u) << "it must actually have waited before giving up";
}

TEST(TopoWorkerFailures, AHealthyWaitNeverReachesTheTimeout)
{
    // The converse, so the case above cannot pass merely because the budget is
    // always exceeded: with the clock stopped, a predecessor that retires is
    // waited for indefinitely and no error is latched.
    Chain2 g;
    g.counters[0].state = TASK_PENDING;
    g.head.next = 1;

    ScriptedPlatform plat;
    plat.budget = 10;
    plat.clock_step_per_spin = 0;  // time does not pass

    // Retire the producer after a few spins, from the execute side of a
    // separate step is not available here -- so drive it by hand: the policy
    // flips the counter once it has spun enough.
    struct FlippingPlatform : ScriptedPlatform {
        TaskCounter *producer = nullptr;
        void spin_hint() {
            ScriptedPlatform::spin_hint();
            if (spins == 5) producer->state = TASK_DONE;
        }
    } flip;
    flip.budget = 10;
    flip.clock_step_per_spin = 0;
    flip.producer = &g.counters[0];

    bool ran = false;
    const WorkerStep step = worker_step(
        flip, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries,
        [&](const TaskEntry &) {
            ran = true;
            return true;
        }
    );

    EXPECT_EQ(step, WorkerStep::RAN_TASK);
    EXPECT_TRUE(ran);
    EXPECT_EQ(g.control.error, TOPO_OK);
    EXPECT_EQ(g.counters[1].state, TASK_DONE);
    EXPECT_EQ(flip.spins, 5u);
}

TEST(TopoWorkerFailures, AFailingTaskPublishesFailedAndLatchesTaskFailed)
{
    // The execute-returns-false path: the counter must reach a terminal state
    // (or consumers spin forever) and the code must say which failure it was.
    Chain2 g;
    ScriptedPlatform plat;

    const WorkerStep step = worker_step(
        plat, &g.head, &g.control, g.order, 2, CORE_TYPE_AIC, g.counters, g.fanin_offsets, g.fanin_ids, g.entries,
        [](const TaskEntry &) { return false; }
    );

    EXPECT_EQ(step, WorkerStep::ABORTED);
    EXPECT_EQ(g.counters[0].state, TASK_FAILED);
    EXPECT_EQ(g.control.error, TOPO_ERR_TASK_FAILED);
}

TEST(TopoWorkerFailures, TheRootCauseWinsTheLatchRaceAgainstItsOwnConsumers)
{
    // worker_step latches TOPO_ERR_TASK_FAILED before publishing FAILED. The
    // other order loses the root cause: a consumer already spinning on the
    // counter sees FAILED the instant it is stored and latches
    // TOPO_ERR_PREDECESSOR_FAILED, so whichever reaches the CAS first wins and
    // the run gets explained by a consequence.
    //
    // Run it many times with the producer and a crowd of consumers live at
    // once; a single pass proves nothing about a race.
    // Deterministic now that the policy holds the window open, so a handful of
    // attempts is enough; each one pays the widened delay.
    for (int attempt = 0; attempt < 20; ++attempt) {
        constexpr uint32_t kConsumers = 6;

        // Task 0 fails; tasks 1..kConsumers all depend on it.
        const uint32_t task_count = kConsumers + 1;
        std::vector<uint32_t> order(task_count);
        std::vector<uint32_t> fanin_offsets(task_count + 1, 0);
        std::vector<uint32_t> fanin_ids;
        std::vector<TaskCounter> counters(task_count);
        std::vector<TaskEntry> entries(task_count);
        for (uint32_t t = 0; t < task_count; ++t) order[t] = t;
        fanin_offsets[0] = 0;
        fanin_offsets[1] = 0;
        for (uint32_t t = 1; t < task_count; ++t) {
            fanin_ids.push_back(0);
            fanin_offsets[t + 1] = static_cast<uint32_t>(fanin_ids.size());
        }

        QueueHead head{};
        RunControl control{};

        // Real atomics here: this case is about a race, so the host policy has
        // to be the concurrent one.
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
                // Widen the window between publishing FAILED and whatever the
                // caller does next. Racing for it is hopeless -- the gap is a
                // few instructions, and 200 unassisted attempts never hit it --
                // so the policy holds it open instead.
                //
                // With the correct order the delay is harmless: the code is
                // already latched before the store, so a consumer that wakes
                // here loses the CAS. With the order reversed it hands every
                // waiting consumer all the time it needs to latch its own
                // consequence first, which is what lets this case fail.
                if (s == TASK_FAILED) {
                    for (int i = 0; i < 100; ++i) std::this_thread::yield();
                }
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

        std::vector<std::thread> threads;
        threads.reserve(task_count);
        for (uint32_t w = 0; w < task_count; ++w) {
            threads.emplace_back([&] {
                AtomicPlatform plat;
                worker_loop(
                    plat, &head, &control, order.data(), task_count, CORE_TYPE_AIC, counters.data(), fanin_offsets.data(),
                    fanin_ids.data(), entries.data(),
                    [](const TaskEntry &e) { return e.task_id != 0; }  // task 0 fails
                );
            });
        }
        for (auto &t : threads) t.join();

        ASSERT_EQ(control.error, TOPO_ERR_TASK_FAILED)
            << "attempt " << attempt << ": the run must be explained by the task that failed, not by a consumer "
            << "reacting to it";
        ASSERT_EQ(counters[0].state, TASK_FAILED);
    }
}
