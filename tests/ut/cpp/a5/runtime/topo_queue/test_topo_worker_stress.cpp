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
 * Randomized coverage of the worker loop, complementing the hand-built cases in
 * test_topo_worker.cpp. Those pin named shapes; these sweep generated DAGs
 * across worker counts and assert the two properties the design rests on: every
 * task runs exactly once, and it runs strictly after every predecessor finished.
 *
 * Task ids are deliberately NOT in topological order -- the generator relabels
 * through a random permutation -- so a case here fails if the loop ever reads
 * `order` as an identity mapping.
 *
 * Every run carries a deadline. A worker that cannot make progress spins with no
 * back-off, so an unbounded join would hang the whole suite rather than report
 * the case; run_with_deadline turns that into a failure with the DAG shape, the
 * worker count, and the counter census attached.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "topo_sort.h"
#include "topo_worker.h"

using namespace simpler::topo_queue;

namespace {

// Host stand-in for the device primitives, matching test_topo_worker.cpp's
// policy. `timeout_ms` is settable so one case can drive the wait budget.
struct HostPlatform {
    uint64_t timeout_ms = 120'000;

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
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }
    uint64_t timeout_ticks() { return timeout_ms; }
    void spin_hint() {}
};

struct Image {
    uint32_t task_count;
    std::vector<uint32_t> fanin_offsets;
    std::vector<uint32_t> fanin_ids;
    std::vector<uint32_t> order;
    std::vector<TaskCounter> counters;
    std::vector<TaskEntry> entries;
    QueueHead head{};
    RunControl control{};
    uint32_t sort_rc = TOPO_OK;

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
        sort_rc = topo_sort(task_count, static_cast<uint32_t>(fanin_ids.size()), fanin_offsets.data(),
                            fanin_ids.data(), s, order.data());
    }
};

struct RunOutcome {
    uint32_t retired = 0;
    bool timed_out = false;
};

/*
 * Drive `worker_count` threads over one image and join them, or fail the case if
 * the run outlives `deadline_ms`. On expiry the run-wide error is latched, which
 * releases every spinning worker: a worker that has pulled a task always
 * publishes a terminal counter, so no peer can be left waiting on it.
 */
template <typename Execute>
RunOutcome run_with_deadline(Image &img, uint32_t worker_count, Execute &execute, const std::string &what,
                             uint64_t deadline_ms = 60'000, uint64_t worker_timeout_ms = 120'000,
                             std::atomic<uint32_t> *workers_started = nullptr) {
    std::vector<std::thread> threads;
    std::atomic<uint32_t> total{0};
    std::atomic<uint32_t> finished{0};
    threads.reserve(worker_count);
    for (uint32_t w = 0; w < worker_count; ++w) {
        threads.emplace_back([&] {
            if (workers_started != nullptr) workers_started->fetch_add(1, std::memory_order_release);
            HostPlatform plat;
            plat.timeout_ms = worker_timeout_ms;
            total.fetch_add(worker_loop(plat, &img.head, &img.control, img.order.data(), img.task_count,
                                        CORE_TYPE_AIV, img.counters.data(), img.fanin_offsets.data(), img.fanin_ids.data(),
                                        img.entries.data(), execute),
                            std::memory_order_relaxed);
            finished.fetch_add(1, std::memory_order_release);
        });
    }

    RunOutcome out;
    const auto start = std::chrono::steady_clock::now();
    std::atomic<bool> joined{false};
    std::thread deadline_watch([&] {
        while (!joined.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            if (static_cast<uint64_t>(elapsed) <= deadline_ms) continue;

            uint32_t pending = 0, done = 0, failed = 0;
            for (uint32_t t = 0; t < img.task_count; ++t) {
                const uint64_t s = img.counters[t].state;
                if (s == TASK_PENDING) ++pending;
                else if (s == TASK_DONE) ++done;
                else ++failed;
            }
            out.timed_out = true;
            ADD_FAILURE() << "run did not finish within " << deadline_ms << " ms: " << what
                          << " workers=" << worker_count << " tasks=" << img.task_count
                          << " workers_finished=" << finished.load() << " head=" << img.head.next
                          << " error=" << img.control.error << " counters{pending=" << pending
                          << " done=" << done << " failed=" << failed << "}";
            HostPlatform rescue;
            rescue.latch_error(&img.control, TOPO_ERR_WAIT_TIMEOUT);
            return;
        }
    });

    for (auto &t : threads) t.join();
    joined.store(true, std::memory_order_release);
    deadline_watch.join();
    out.retired = total.load();
    return out;
}

// Edges are generated over positions (always low -> high, so acyclic by
// construction) and then relabelled, so the task ids a case runs on are in no
// particular order.
struct Dag {
    std::vector<std::vector<uint32_t>> fanin;
    std::string shape;
};

Dag make_dag(uint64_t seed) {
    std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + 12345);
    uint32_t n = 1 + static_cast<uint32_t>(rng() % 220);
    std::vector<std::vector<uint32_t>> by_pos(n);
    char label[96];

    switch (rng() % 8) {
        case 0:
            for (uint32_t i = 1; i < n; ++i) by_pos[i].push_back(i - 1);
            snprintf(label, sizeof(label), "chain(n=%u)", n);
            break;
        case 1: {
            static const double kDensity[] = {0.01, 0.05, 0.25, 0.75};
            const double p = kDensity[rng() % 4];
            std::bernoulli_distribution coin(p);
            for (uint32_t j = 1; j < n; ++j)
                for (uint32_t i = 0; i < j; ++i)
                    if (coin(rng)) by_pos[j].push_back(i);
            snprintf(label, sizeof(label), "random(n=%u,p=%.2f)", n, p);
            break;
        }
        case 2: {
            const uint32_t width = 1 + static_cast<uint32_t>(rng() % 10);
            const uint32_t layers = std::max(1u, n / width);
            n = width * layers;
            by_pos.assign(n, {});
            for (uint32_t l = 1; l < layers; ++l)
                for (uint32_t i = 0; i < width; ++i)
                    for (uint32_t p = 0; p < width; ++p) by_pos[l * width + i].push_back((l - 1) * width + p);
            snprintf(label, sizeof(label), "layered(layers=%u,width=%u)", layers, width);
            break;
        }
        case 3: {
            const uint32_t k = 2 + static_cast<uint32_t>(rng() % 6);
            const uint32_t per = std::max(1u, n / k);
            n = per * k;
            by_pos.assign(n, {});
            for (uint32_t c = 0; c < k; ++c)
                for (uint32_t i = 1; i < per; ++i)
                    by_pos[c * per + i].push_back(c * per + i - 1 - static_cast<uint32_t>(rng() % i));
            snprintf(label, sizeof(label), "components(k=%u,per=%u)", k, per);
            break;
        }
        case 4:
            n = std::max(4u, (n / 4) * 4);
            by_pos.assign(n, {});
            for (uint32_t b = 0; b + 4 <= n; b += 4) {
                by_pos[b + 1].push_back(b);
                by_pos[b + 2].push_back(b);
                by_pos[b + 3].push_back(b + 1);
                by_pos[b + 3].push_back(b + 2);
                if (b + 4 < n) by_pos[b + 4].push_back(b + 3);
            }
            snprintf(label, sizeof(label), "diamonds(n=%u)", n);
            break;
        case 5:
            for (uint32_t i = 0; i + 1 < n; ++i) by_pos[n - 1].push_back(i);
            snprintf(label, sizeof(label), "single_sink(n=%u)", n);
            break;
        case 6:
            for (uint32_t i = 1; i < n; ++i) by_pos[i].push_back(0);
            snprintf(label, sizeof(label), "single_source(n=%u)", n);
            break;
        default:
            snprintf(label, sizeof(label), "independent(n=%u)", n);
            break;
    }

    std::vector<uint32_t> relabel(n);
    for (uint32_t i = 0; i < n; ++i) relabel[i] = i;
    std::shuffle(relabel.begin(), relabel.end(), rng);

    Dag d;
    d.shape = label;
    d.fanin.assign(n, {});
    for (uint32_t p = 0; p < n; ++p) {
        auto &row = d.fanin[relabel[p]];
        for (uint32_t q : by_pos[p]) row.push_back(relabel[q]);
        std::shuffle(row.begin(), row.end(), rng);  // a CSR row need not be sorted
    }
    return d;
}

const uint32_t kWorkerCounts[] = {1, 2, 4, 8, 16, 64};

}  // namespace

TEST(TopoWorkerStress, RandomDagsRunEveryTaskExactlyOnceAfterItsPredecessors) {
    constexpr uint64_t kDags = 32;
    for (uint64_t seed = 0; seed < kDags; ++seed) {
        const Dag dag = make_dag(seed);
        const uint32_t n = static_cast<uint32_t>(dag.fanin.size());
        for (uint32_t workers : kWorkerCounts) {
            Image img(dag.fanin);
            ASSERT_EQ(img.sort_rc, TOPO_OK) << dag.shape;
            std::vector<uint32_t> position(n);
            ASSERT_TRUE(order_is_topological(n, img.fanin_offsets.data(), img.fanin_ids.data(),
                                             img.order.data(), position.data()))
                << dag.shape;

            std::vector<std::atomic<uint32_t>> runs(n);
            std::vector<std::atomic<uint32_t>> finished(n);
            for (uint32_t t = 0; t < n; ++t) {
                runs[t].store(0);
                finished[t].store(0);
            }
            std::atomic<uint32_t> ran_too_early{0};
            std::atomic<uint32_t> pred_counter_not_done{0};

            auto execute = [&](const TaskEntry &e) {
                const uint32_t t = e.task_id;
                for (uint32_t x = img.fanin_offsets[t]; x < img.fanin_offsets[t + 1]; ++x) {
                    const uint32_t p = img.fanin_ids[x];
                    // Two independent readings of one property: the counter is
                    // what the wait loop itself gates on, the flag is what the
                    // producer's execute actually did. They disagree only if the
                    // release/acquire edge between the two has been lost.
                    auto *a = reinterpret_cast<const std::atomic<uint64_t> *>(
                        const_cast<const uint64_t *>(&img.counters[p].state));
                    if (a->load(std::memory_order_acquire) != TASK_DONE) {
                        pred_counter_not_done.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (finished[p].load(std::memory_order_acquire) != 1u) {
                        ran_too_early.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                runs[t].fetch_add(1, std::memory_order_relaxed);
                finished[t].store(1, std::memory_order_release);
                return true;
            };

            const RunOutcome out = run_with_deadline(img, workers, execute, dag.shape);
            if (out.timed_out) return;

            EXPECT_EQ(img.control.error, TOPO_OK) << dag.shape << " workers=" << workers;
            EXPECT_EQ(out.retired, n) << dag.shape << " workers=" << workers;
            EXPECT_EQ(ran_too_early.load(), 0u) << dag.shape << " workers=" << workers;
            EXPECT_EQ(pred_counter_not_done.load(), 0u) << dag.shape << " workers=" << workers;
            for (uint32_t t = 0; t < n; ++t) {
                ASSERT_EQ(runs[t].load(), 1u) << dag.shape << " workers=" << workers << " task=" << t;
                ASSERT_EQ(img.counters[t].state, TASK_DONE) << dag.shape << " workers=" << workers
                                                            << " task=" << t;
            }
        }
    }
}

TEST(TopoWorkerStress, ChainDeeperThanTheWorkerPoolStillDrains) {
    // Far fewer workers than the longest dependency chain: all but one worker is
    // blocked at any moment, and the queue order is what keeps the unblocked one
    // from being a different worker each time.
    constexpr uint32_t kLen = 500;
    for (uint32_t workers : {1u, 2u, 3u, 5u}) {
        std::vector<std::vector<uint32_t>> fanin(kLen);
        for (uint32_t t = 1; t < kLen; ++t) fanin[t].push_back(t - 1);
        Image img(fanin);
        ASSERT_EQ(img.sort_rc, TOPO_OK);
        auto execute = [](const TaskEntry &) { return true; };
        const RunOutcome out = run_with_deadline(img, workers, execute, "chain(500)");
        if (out.timed_out) return;
        EXPECT_EQ(out.retired, kLen) << "workers=" << workers;
        EXPECT_EQ(img.control.error, TOPO_OK) << "workers=" << workers;
    }
}

TEST(TopoWorkerStress, FarMoreWorkersThanTasksDrainWithoutSpinning) {
    for (uint32_t tasks : {1u, 2u, 7u}) {
        std::vector<std::vector<uint32_t>> fanin(tasks);
        for (uint32_t t = 1; t < tasks; ++t) fanin[t].push_back(t - 1);
        Image img(fanin);
        ASSERT_EQ(img.sort_rc, TOPO_OK);
        auto execute = [](const TaskEntry &) { return true; };
        const RunOutcome out = run_with_deadline(img, 64, execute, "tiny graph, 64 workers");
        if (out.timed_out) return;
        EXPECT_EQ(out.retired, tasks);
        EXPECT_EQ(img.control.error, TOPO_OK);
        // A worker learns the queue is drained only BY pulling an index past the
        // end, so the head necessarily overshoots. Every slot must have been
        // handed out, and each worker may overshoot at most once -- a head above
        // that bound would mean a worker pulled again after being told to stop.
        EXPECT_GE(img.head.next, tasks);
        EXPECT_LE(img.head.next, tasks + 64u);
    }
}

TEST(TopoWorkerStress, AFailedTaskEndsTheRunAndNeverStrandsAPeer) {
    // A failing task must terminate the run through the error latch and the
    // FAILED counter, not through the wait budget: a peer that spun out its
    // budget would report TOPO_ERR_WAIT_TIMEOUT instead.
    for (uint64_t seed = 0; seed < 16; ++seed) {
        const Dag dag = make_dag(seed);
        const uint32_t n = static_cast<uint32_t>(dag.fanin.size());
        std::mt19937_64 rng(seed ^ 0xA5A5A5A5ull);
        const uint32_t victim = static_cast<uint32_t>(rng() % n);
        for (uint32_t workers : {1u, 4u, 16u}) {
            Image img(dag.fanin);
            ASSERT_EQ(img.sort_rc, TOPO_OK) << dag.shape;
            std::vector<std::atomic<uint32_t>> runs(n);
            std::vector<std::atomic<uint32_t>> finished(n);
            for (uint32_t t = 0; t < n; ++t) {
                runs[t].store(0);
                finished[t].store(0);
            }
            std::atomic<uint32_t> ran_too_early{0};
            std::atomic<uint32_t> pred_counter_not_done{0};

            auto execute = [&](const TaskEntry &e) {
                const uint32_t t = e.task_id;
                for (uint32_t x = img.fanin_offsets[t]; x < img.fanin_offsets[t + 1]; ++x) {
                    const uint32_t p = img.fanin_ids[x];
                    auto *a = reinterpret_cast<const std::atomic<uint64_t> *>(
                        const_cast<const uint64_t *>(&img.counters[p].state));
                    if (a->load(std::memory_order_acquire) != TASK_DONE) {
                        pred_counter_not_done.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (finished[p].load(std::memory_order_acquire) != 1u) {
                        ran_too_early.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                runs[t].fetch_add(1, std::memory_order_relaxed);
                if (t == victim) return false;
                finished[t].store(1, std::memory_order_release);
                return true;
            };

            const RunOutcome out = run_with_deadline(img, workers, execute, dag.shape + "/failure");
            if (out.timed_out) return;

            EXPECT_NE(img.control.error, TOPO_OK) << dag.shape << " victim=" << victim;
            EXPECT_NE(img.control.error, TOPO_ERR_WAIT_TIMEOUT)
                << dag.shape << " victim=" << victim << ": a peer spun out the wait budget";
            EXPECT_EQ(img.counters[victim].state, TASK_FAILED) << dag.shape << " victim=" << victim;
            EXPECT_EQ(ran_too_early.load(), 0u) << dag.shape << " victim=" << victim
                                                << " workers=" << workers;
            EXPECT_EQ(pred_counter_not_done.load(), 0u)
                << dag.shape << " victim=" << victim << " workers=" << workers;
            for (uint32_t t = 0; t < n; ++t) {
                ASSERT_LE(runs[t].load(), 1u) << dag.shape << " task=" << t;
                if (img.counters[t].state == TASK_DONE) {
                    ASSERT_EQ(runs[t].load(), 1u) << dag.shape << " task " << t << " is DONE but never ran";
                }
            }
        }
    }
}

TEST(TopoWorkerStress, RepeatedRunsOfTheTightestShapeStayOrdered) {
    // The shape with the least slack: one source, one sink, and a middle layer
    // wide enough that every worker is either running or waiting. Repetition is
    // the point -- a rare interleaving needs many attempts, not a bigger graph.
    constexpr uint32_t kMiddle = 12;
    std::vector<std::vector<uint32_t>> fanin(kMiddle + 2);
    for (uint32_t i = 1; i <= kMiddle; ++i) {
        fanin[i].push_back(0);
        fanin[kMiddle + 1].push_back(i);
    }
    for (uint32_t attempt = 0; attempt < 150; ++attempt) {
        Image img(fanin);
        ASSERT_EQ(img.sort_rc, TOPO_OK);
        const uint32_t n = img.task_count;
        std::vector<std::atomic<uint32_t>> runs(n);
        std::vector<std::atomic<uint32_t>> finished(n);
        for (uint32_t t = 0; t < n; ++t) {
            runs[t].store(0);
            finished[t].store(0);
        }
        std::atomic<uint32_t> ran_too_early{0};
        std::atomic<uint32_t> pred_counter_not_done{0};
        auto execute = [&](const TaskEntry &e) {
            const uint32_t t = e.task_id;
            for (uint32_t x = img.fanin_offsets[t]; x < img.fanin_offsets[t + 1]; ++x) {
                const uint32_t p = img.fanin_ids[x];
                auto *a = reinterpret_cast<const std::atomic<uint64_t> *>(
                    const_cast<const uint64_t *>(&img.counters[p].state));
                if (a->load(std::memory_order_acquire) != TASK_DONE) {
                    pred_counter_not_done.fetch_add(1, std::memory_order_relaxed);
                }
                if (finished[p].load(std::memory_order_acquire) != 1u) {
                    ran_too_early.fetch_add(1, std::memory_order_relaxed);
                }
            }
            runs[t].fetch_add(1, std::memory_order_relaxed);
            finished[t].store(1, std::memory_order_release);
            return true;
        };
        const RunOutcome out = run_with_deadline(img, 8, execute, "source-fan-sink");
        if (out.timed_out) return;
        ASSERT_EQ(out.retired, n) << "attempt " << attempt;
        ASSERT_EQ(ran_too_early.load(), 0u) << "attempt " << attempt;
        ASSERT_EQ(pred_counter_not_done.load(), 0u) << "attempt " << attempt;
        for (uint32_t t = 0; t < n; ++t) ASSERT_EQ(runs[t].load(), 1u) << "attempt " << attempt << " task " << t;
    }
}

TEST(TopoWorkerStress, TheWaitBudgetCoversTheInFlightWindowNotOneProducer) {
    // The deadline is taken once, when a worker starts waiting, and a worker
    // claims its index as soon as the head hands one out. So a worker can sit on
    // an index while up to worker_count - 1 tasks ahead of it are still running,
    // and the budget has to cover that whole window draining rather than the
    // single producer the waiting task names.
    //
    // Both halves run the same chain with the same 1 ms budget and the same slow
    // task. The only difference is how many workers there are.
    constexpr uint32_t kLen = 8;
    std::vector<std::vector<uint32_t>> fanin(kLen);
    for (uint32_t t = 1; t < kLen; ++t) fanin[t].push_back(t - 1);

    // One worker pulls index i only after retiring i - 1, so it never enters the
    // wait loop with a pending predecessor and no budget can expire -- however
    // slow the tasks are and however long the queue is.
    {
        Image serial(fanin);
        ASSERT_EQ(serial.sort_rc, TOPO_OK);
        std::atomic<uint32_t> executed{0};
        auto slow = [&](const TaskEntry &) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
            while (std::chrono::steady_clock::now() < until) {
            }
            executed.fetch_add(1, std::memory_order_relaxed);
            return true;
        };
        const RunOutcome out = run_with_deadline(serial, 1, slow, "chain(8) serial", 60'000, 1);
        if (out.timed_out) return;
        EXPECT_EQ(serial.control.error, TOPO_OK);
        EXPECT_EQ(out.retired, kLen);
        EXPECT_EQ(executed.load(), kLen);
    }

    // Four workers over the same chain hold indices 1..3 while index 0 runs, so
    // three of them are in a wait that outlives a 1 ms budget. The first task
    // holds until every worker is up, which is what makes the wait happen at all
    // rather than depending on thread start order.
    {
        Image parallel(fanin);
        ASSERT_EQ(parallel.sort_rc, TOPO_OK);
        constexpr uint32_t kWorkers = 4;
        std::atomic<uint32_t> workers_started{0};
        std::atomic<uint32_t> executed{0};
        std::vector<std::atomic<uint32_t>> ran(kLen);
        for (uint32_t t = 0; t < kLen; ++t) ran[t].store(0);

        auto gated = [&](const TaskEntry &e) {
            if (e.task_id == 0) {
                while (workers_started.load(std::memory_order_acquire) < kWorkers) {
                }
                const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                while (std::chrono::steady_clock::now() < until) {
                }
            }
            ran[e.task_id].store(1, std::memory_order_relaxed);
            executed.fetch_add(1, std::memory_order_relaxed);
            return true;
        };

        const RunOutcome out =
            run_with_deadline(parallel, kWorkers, gated, "chain(8) parallel", 60'000, 1, &workers_started);
        if (out.timed_out) return;

        EXPECT_EQ(parallel.control.error, TOPO_ERR_WAIT_TIMEOUT)
            << "nothing in this run failed; the verdict comes from the wait budget alone";
        EXPECT_LT(executed.load(), kLen);
        uint32_t poisoned_without_running = 0;
        for (uint32_t t = 0; t < kLen; ++t) {
            if (parallel.counters[t].state == TASK_FAILED && ran[t].load() == 0u) ++poisoned_without_running;
        }
        EXPECT_GT(poisoned_without_running, 0u) << "a timed-out worker publishes FAILED for a task that "
                                                   "never ran";
    }
}
