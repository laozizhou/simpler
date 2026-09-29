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
 * Drives the real worker loop with a host platform policy: std::atomic stands in
 * for the GM primitives, std::thread for the AICore workers. What this covers is
 * the algorithm -- ordering, the counter state machine, failure propagation,
 * drain. What it cannot cover is cache-line write-back behaviour, which has no
 * host equivalent; that stays a silicon question.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "topo_sort.h"
#include "topo_worker.h"

using namespace simpler::topo_queue;

namespace {

// Host stand-in for the device primitives. Each call maps to the ordering the
// device policy owes, so a test that passes here has at least exercised the
// same happens-before edges the silicon build must provide.
struct HostPlatform {
    std::atomic<uint64_t> spins{0};

    uint64_t load_queue_head(const QueueHead *head) {
        auto *a = reinterpret_cast<const std::atomic<uint64_t> *>(const_cast<const uint64_t *>(&head->next));
        return a->load(std::memory_order_acquire);
    }
    bool try_claim_queue_head(QueueHead *head, uint64_t expected) {
        auto *a = reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(&head->next));
        return a->compare_exchange_strong(expected, expected + 1, std::memory_order_acq_rel);
    }

    uint64_t load_counter_acquire(const TaskCounter *c) {
        auto *a = reinterpret_cast<const std::atomic<uint64_t> *>(const_cast<const uint64_t *>(&c->state));
        return a->load(std::memory_order_acquire);
    }

    void store_counter_release(TaskCounter *c, uint64_t state) {
        auto *a = reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(&c->state));
        a->store(state, std::memory_order_release);
    }

    uint64_t load_error_relaxed(const RunControl *rc) {
        auto *a = reinterpret_cast<const std::atomic<uint64_t> *>(const_cast<const uint64_t *>(&rc->error));
        return a->load(std::memory_order_relaxed);
    }

    void latch_error(RunControl *rc, uint64_t code) {
        auto *a = reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(&rc->error));
        uint64_t expected = TOPO_OK;
        a->compare_exchange_strong(expected, code, std::memory_order_acq_rel);
    }

    uint64_t now_ticks() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    // Generous: these tests must never hit it on a healthy run, and a loaded CI
    // box is slow. A case that wants the timeout drives it explicitly.
    uint64_t timeout_ticks() { return 10'000; }

    void spin_hint() { spins.fetch_add(1, std::memory_order_relaxed); }
};

// One run's GM image, host-side.
struct Image {
    uint32_t task_count;
    std::vector<uint32_t> fanin_offsets;
    std::vector<uint32_t> fanin_ids;
    std::vector<uint32_t> order;
    std::vector<TaskCounter> counters;
    std::vector<TaskEntry> entries;
    QueueHead head{};
    RunControl control{};

    explicit Image(const std::vector<std::vector<uint32_t>> &fanin) :
        task_count(static_cast<uint32_t>(fanin.size())) {
        fanin_offsets.push_back(0);
        for (const auto &row : fanin) {
            for (uint32_t p : row) fanin_ids.push_back(p);
            fanin_offsets.push_back(static_cast<uint32_t>(fanin_ids.size()));
        }
        order.resize(task_count);
        counters.assign(task_count, TaskCounter{});
        entries.resize(task_count);
        for (uint32_t t = 0; t < task_count; ++t) {
            entries[t].task_id = t;
            entries[t].core_type = CORE_TYPE_AIV;
        }

        std::vector<uint32_t> in_degree(task_count);
        std::vector<uint32_t> fanout_offsets(task_count + 1);
        std::vector<uint32_t> fanout_ids(fanin_ids.size());
        std::vector<uint32_t> frontier(task_count);
        TopoSortScratch s{in_degree.data(), fanout_offsets.data(), fanout_ids.data(), frontier.data()};
        const uint32_t rc = topo_sort(
            task_count, static_cast<uint32_t>(fanin_ids.size()), fanin_offsets.data(), fanin_ids.data(), s,
            order.data()
        );
        EXPECT_EQ(rc, TOPO_OK);
        head.next = 0;
        control.error = TOPO_OK;
        control.retired = 0;
    }
};

// Runs `worker_count` threads over one image. `execute` sees every task.
template <typename Execute>
uint32_t run_workers(Image &img, uint32_t worker_count, Execute execute) {
    std::vector<std::thread> threads;
    std::atomic<uint32_t> total{0};
    threads.reserve(worker_count);
    for (uint32_t w = 0; w < worker_count; ++w) {
        threads.emplace_back([&] {
            HostPlatform plat;
            const uint32_t retired = worker_loop(
                plat, &img.head, &img.control, img.order.data(), img.task_count, CORE_TYPE_AIV, img.counters.data(),
                img.fanin_offsets.data(), img.fanin_ids.data(), img.entries.data(), execute
            );
            total.fetch_add(retired, std::memory_order_relaxed);
        });
    }
    for (auto &t : threads) t.join();
    return total.load();
}

}  // namespace

TEST(TopoWorker, SingleWorkerRunsEveryTaskInOrder) {
    Image img({{}, {0}, {1}, {2}});
    std::vector<uint32_t> seen;
    const uint32_t retired = run_workers(img, 1, [&](const TaskEntry &e) {
        seen.push_back(e.task_id);
        return true;
    });
    EXPECT_EQ(retired, 4u);
    ASSERT_EQ(seen.size(), 4u);
    for (uint32_t i = 0; i < 4; ++i) EXPECT_EQ(seen[i], i);
    EXPECT_EQ(img.control.error, TOPO_OK);
}

TEST(TopoWorker, EveryTaskRunsExactlyOnce) {
    // 8 workers over a diamond-of-diamonds. The queue head is the only thing
    // handing out work, so a double-run would mean the fetch_add is not atomic
    // or the drain test is off by one.
    Image img({{}, {0}, {0}, {1, 2}, {3}, {3}, {4, 5}});
    std::vector<std::atomic<uint32_t>> runs(img.task_count);
    for (auto &r : runs) r.store(0);

    const uint32_t retired = run_workers(img, 8, [&](const TaskEntry &e) {
        runs[e.task_id].fetch_add(1, std::memory_order_relaxed);
        return true;
    });

    EXPECT_EQ(retired, img.task_count);
    for (uint32_t t = 0; t < img.task_count; ++t) EXPECT_EQ(runs[t].load(), 1u) << "task " << t;
}

TEST(TopoWorker, PredecessorsAlwaysCompleteFirst) {
    // The property the whole design rests on: when a task runs, every
    // predecessor's counter already reads DONE. Checked from inside execute, so
    // a violation fails the case rather than merely producing a wrong answer.
    constexpr uint32_t kLayers = 6;
    constexpr uint32_t kWidth = 5;
    std::vector<std::vector<uint32_t>> fanin(kLayers * kWidth);
    for (uint32_t layer = 1; layer < kLayers; ++layer) {
        for (uint32_t i = 0; i < kWidth; ++i) {
            for (uint32_t p = 0; p < kWidth; ++p) {
                fanin[layer * kWidth + i].push_back((layer - 1) * kWidth + p);
            }
        }
    }
    Image img(fanin);
    std::atomic<uint32_t> violations{0};

    const uint32_t retired = run_workers(img, 8, [&](const TaskEntry &e) {
        const uint32_t begin = img.fanin_offsets[e.task_id];
        const uint32_t end = img.fanin_offsets[e.task_id + 1];
        for (uint32_t x = begin; x < end; ++x) {
            const uint32_t producer = img.fanin_ids[x];
            auto *a = reinterpret_cast<const std::atomic<uint64_t> *>(
                const_cast<const uint64_t *>(&img.counters[producer].state)
            );
            if (a->load(std::memory_order_acquire) != TASK_DONE) violations.fetch_add(1);
        }
        return true;
    });

    EXPECT_EQ(retired, img.task_count);
    EXPECT_EQ(violations.load(), 0u);
}

TEST(TopoWorker, MoreWorkersThanTasksDrainsCleanly) {
    // Workers that find the queue already empty must return, not spin. With 16
    // workers and 3 tasks, 13 of them take the drain path immediately.
    Image img({{}, {0}, {1}});
    const uint32_t retired = run_workers(img, 16, [](const TaskEntry &) { return true; });
    EXPECT_EQ(retired, 3u);
    EXPECT_EQ(img.control.error, TOPO_OK);
}

TEST(TopoWorker, LongChainWithManyWorkersStillCompletes) {
    // The pathological occupancy case: a 200-task chain with 8 workers means 7
    // of them are blocked at any moment. It must still finish -- that is the
    // deadlock-freedom argument in its least comfortable shape.
    constexpr uint32_t kLen = 200;
    std::vector<std::vector<uint32_t>> fanin(kLen);
    for (uint32_t t = 1; t < kLen; ++t) fanin[t].push_back(t - 1);
    Image img(fanin);

    const uint32_t retired = run_workers(img, 8, [](const TaskEntry &) { return true; });
    EXPECT_EQ(retired, kLen);
    for (uint32_t t = 0; t < kLen; ++t) EXPECT_EQ(img.counters[t].state, TASK_DONE) << "task " << t;
}

TEST(TopoWorker, FailedTaskLatchesErrorAndStopsTheRun) {
    Image img({{}, {0}, {1}, {2}});
    std::atomic<uint32_t> ran{0};

    run_workers(img, 4, [&](const TaskEntry &e) {
        ran.fetch_add(1, std::memory_order_relaxed);
        return e.task_id != 1;  // task 1 fails
    });

    EXPECT_NE(img.control.error, TOPO_OK);
    EXPECT_EQ(img.counters[1].state, TASK_FAILED);
    // 3 must not have run: its predecessor chain is poisoned.
    EXPECT_NE(img.counters[3].state, TASK_DONE);
}

TEST(TopoWorker, ErrorLatchIsTheAuthoritativeFailureSignal) {
    // Poisoning a consumer's counter is best-effort, not a guarantee: a worker
    // that sees the run-wide error returns instead of pulling further tasks, so
    // a consumer nobody pulled stays PENDING. That is deliberate -- exiting fast
    // beats walking the rest of the graph to mark it -- but it means the error
    // latch, not the counters, is what a reader must consult to decide a run
    // failed. A task's counter only ever answers "did *this* task retire".
    Image img({{}, {0}, {1}});
    run_workers(img, 1, [&](const TaskEntry &e) { return e.task_id != 0; });

    EXPECT_EQ(img.control.error, TOPO_ERR_TASK_FAILED);
    EXPECT_EQ(img.counters[0].state, TASK_FAILED);
    // 1 and 2 were never pulled, so they are untouched rather than poisoned.
    EXPECT_NE(img.counters[1].state, TASK_DONE);
    EXPECT_NE(img.counters[2].state, TASK_DONE);
}

TEST(TopoWorker, AFailedTaskEndsTheRunAndNoConsumerOfItCompletes) {
    // What this case actually holds: a failure terminates the run promptly and
    // nothing downstream of it is ever marked DONE.
    //
    // It does NOT reliably exercise the poison branch, despite the shape below
    // aiming at it. Measured over 300 runs, the sort puts task 3 at index 1 and
    // task 1 at index 2, so the free worker always pulls task 1, fails it, and
    // aborts -- nobody ever pulls task 2, which therefore stays PENDING rather
    // than FAILED. The branch is reached only in a race this test cannot
    // schedule, so it is covered deterministically through the Platform seam in
    // test_topo_worker_failures.cpp instead. Naming this case after the poison
    // path made a vacuous assertion look like a guarantee.
    Image img({{}, {0}, {1}, {}});
    std::atomic<bool> allow_finish{false};

    std::vector<std::thread> threads;
    std::atomic<uint32_t> total{0};
    for (uint32_t w = 0; w < 2; ++w) {
        threads.emplace_back([&] {
            HostPlatform plat;
            total.fetch_add(worker_loop(
                plat, &img.head, &img.control, img.order.data(), img.task_count, CORE_TYPE_AIV, img.counters.data(),
                img.fanin_offsets.data(), img.fanin_ids.data(), img.entries.data(),
                [&](const TaskEntry &e) {
                    if (e.task_id == 3) {
                        // Hold one worker inside execute so the other is the one
                        // that pulls task 2 after task 1 has failed.
                        while (!allow_finish.load(std::memory_order_acquire)) {
                            std::this_thread::yield();
                        }
                    }
                    return e.task_id != 1;
                }
            ));
        });
    }
    // Let the holder go once task 1 has had time to fail.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    allow_finish.store(true, std::memory_order_release);
    for (auto &t : threads) t.join();

    EXPECT_NE(img.control.error, TOPO_OK);
    EXPECT_EQ(img.counters[1].state, TASK_FAILED);
    // Task 2 either was never pulled, or was pulled and poisoned -- never DONE.
    EXPECT_NE(img.counters[2].state, TASK_DONE);
}

TEST(TopoWorker, WideFaninJoinWaitsForAllProducers) {
    // 64 producers into one consumer, 8 workers. The consumer's worker keeps a
    // monotone cursor, so it re-reads only the one producer still blocking --
    // this is the shape that would show a quadratic blow-up if that cursor
    // were ever dropped and the whole row rescanned each pass.
    constexpr uint32_t kProducers = 64;
    std::vector<std::vector<uint32_t>> fanin(kProducers + 1);
    for (uint32_t p = 0; p < kProducers; ++p) fanin[kProducers].push_back(p);
    Image img(fanin);

    std::atomic<uint32_t> producers_done{0};
    std::atomic<bool> consumer_saw_all{false};

    const uint32_t retired = run_workers(img, 8, [&](const TaskEntry &e) {
        if (e.task_id < kProducers) {
            producers_done.fetch_add(1, std::memory_order_acq_rel);
        } else {
            consumer_saw_all.store(producers_done.load(std::memory_order_acquire) == kProducers);
        }
        return true;
    });

    EXPECT_EQ(retired, kProducers + 1);
    EXPECT_TRUE(consumer_saw_all.load());
}

TEST(TopoWorker, EmptyGraphIsANoOp) {
    Image img({});
    const uint32_t retired = run_workers(img, 4, [](const TaskEntry &) { return true; });
    EXPECT_EQ(retired, 0u);
    EXPECT_EQ(img.control.error, TOPO_OK);
}

