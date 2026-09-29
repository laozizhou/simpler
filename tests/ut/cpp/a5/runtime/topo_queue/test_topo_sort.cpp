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

#include <gtest/gtest.h>

#include <vector>

#include "topo_sort.h"

using namespace simpler::topo_queue;

namespace {

// Holds a fanin CSR plus the scratch topo_sort needs, so a case states only its
// edges. Sized exactly, to catch an off-by-one in the offsets arithmetic rather
// than absorbing it into slack.
struct Graph {
    uint32_t task_count;
    std::vector<uint32_t> fanin_offsets;  // task_count + 1
    std::vector<uint32_t> fanin_ids;

    std::vector<uint32_t> in_degree;
    std::vector<uint32_t> fanout_offsets;
    std::vector<uint32_t> fanout_ids;
    std::vector<uint32_t> frontier;
    std::vector<uint32_t> order;
    std::vector<uint32_t> position;

    // `fanin[t]` lists the predecessors of task t.
    explicit Graph(const std::vector<std::vector<uint32_t>> &fanin) : task_count(static_cast<uint32_t>(fanin.size())) {
        fanin_offsets.push_back(0);
        for (const auto &row : fanin) {
            for (uint32_t p : row) fanin_ids.push_back(p);
            fanin_offsets.push_back(static_cast<uint32_t>(fanin_ids.size()));
        }
        in_degree.resize(task_count);
        fanout_offsets.resize(task_count + 1);
        fanout_ids.resize(fanin_ids.size());
        frontier.resize(task_count);
        order.resize(task_count);
        position.resize(task_count);
    }

    TopoSortScratch scratch() {
        return TopoSortScratch{in_degree.data(), fanout_offsets.data(), fanout_ids.data(), frontier.data()};
    }

    uint32_t sort() {
        auto s = scratch();
        return topo_sort(
            task_count, static_cast<uint32_t>(fanin_ids.size()), fanin_offsets.data(), fanin_ids.data(), s,
            order.data()
        );
    }

    bool ordered() {
        return order_is_topological(task_count, fanin_offsets.data(), fanin_ids.data(), order.data(), position.data());
    }
};

}  // namespace

TEST(TopoSort, EmptyGraphSucceeds) {
    Graph g({});
    EXPECT_EQ(g.sort(), TOPO_OK);
}

TEST(TopoSort, SingleTaskNoEdges) {
    Graph g({{}});
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_EQ(g.order[0], 0u);
    EXPECT_TRUE(g.ordered());
}

TEST(TopoSort, IndependentTasksAllEmitted) {
    Graph g({{}, {}, {}, {}});
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_TRUE(g.ordered());
    std::vector<bool> seen(4, false);
    for (uint32_t t : g.order) seen[t] = true;
    for (bool s : seen) EXPECT_TRUE(s);
}

TEST(TopoSort, LinearChainKeepsOrder) {
    // 0 -> 1 -> 2 -> 3
    Graph g({{}, {0}, {1}, {2}});
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_TRUE(g.ordered());
    for (uint32_t i = 0; i < 4; ++i) EXPECT_EQ(g.order[i], i);
}

TEST(TopoSort, DiamondOrdersBothBranchesBeforeJoin) {
    //     0
    //    / \    1   2
    //    \ /
    //     3
    Graph g({{}, {0}, {0}, {1, 2}});
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_TRUE(g.ordered());
    EXPECT_EQ(g.order[0], 0u);
    EXPECT_EQ(g.order[3], 3u);
}

TEST(TopoSort, WideFaninJoinIsOrdered) {
    // 32 independent producers, one consumer that waits on all of them. This is
    // the shape that stresses the worker's per-spin fanin scan.
    std::vector<std::vector<uint32_t>> fanin(33);
    for (uint32_t p = 0; p < 32; ++p) fanin[32].push_back(p);
    Graph g(fanin);
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_TRUE(g.ordered());
    EXPECT_EQ(g.order[32], 32u);
}

TEST(TopoSort, DuplicateEdgeIsOrderedAndCounted) {
    // The host may emit the same producer twice for one consumer. It must not
    // leave the consumer permanently short an in-degree decrement.
    Graph g({{}, {0, 0}});
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_TRUE(g.ordered());
    EXPECT_EQ(g.order[0], 0u);
    EXPECT_EQ(g.order[1], 1u);
}

TEST(TopoSort, SelfLoopIsACycle) {
    Graph g({{0}});
    EXPECT_EQ(g.sort(), TOPO_ERR_CYCLE);
}

TEST(TopoSort, TwoNodeCycleIsRejected) {
    Graph g({{1}, {0}});
    EXPECT_EQ(g.sort(), TOPO_ERR_CYCLE);
}

TEST(TopoSort, CycleBehindAValidPrefixIsRejected) {
    // 0 is fine; 1 and 2 form a cycle. Kahn drains 0 and then stalls, which the
    // emitted-count test has to catch.
    Graph g({{}, {2}, {1}});
    EXPECT_EQ(g.sort(), TOPO_ERR_CYCLE);
}

TEST(TopoSort, OutOfRangePredecessorIsRejected) {
    // A malformed CSR must be refused rather than corrupting the fanout fill.
    Graph g({{}, {7}});
    EXPECT_EQ(g.sort(), TOPO_ERR_CYCLE);
}

TEST(TopoSort, LayeredGraphOrdersEveryLayer) {
    // 8 layers of 8, every task depending on the whole previous layer: 448 edges,
    // enough for the prefix-sum and the cursor restore to be wrong in a way a
    // small graph would hide.
    constexpr uint32_t kLayers = 8;
    constexpr uint32_t kWidth = 8;
    std::vector<std::vector<uint32_t>> fanin(kLayers * kWidth);
    for (uint32_t layer = 1; layer < kLayers; ++layer) {
        for (uint32_t i = 0; i < kWidth; ++i) {
            const uint32_t self = layer * kWidth + i;
            for (uint32_t p = 0; p < kWidth; ++p) fanin[self].push_back((layer - 1) * kWidth + p);
        }
    }
    Graph g(fanin);
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_TRUE(g.ordered());

    // Every task of layer L must precede every task of layer L+1.
    std::vector<uint32_t> pos(kLayers * kWidth);
    for (uint32_t i = 0; i < g.task_count; ++i) pos[g.order[i]] = i;
    for (uint32_t layer = 1; layer < kLayers; ++layer) {
        for (uint32_t i = 0; i < kWidth; ++i) {
            for (uint32_t p = 0; p < kWidth; ++p) {
                EXPECT_LT(pos[(layer - 1) * kWidth + p], pos[layer * kWidth + i]);
            }
        }
    }
}

TEST(TopoSort, FanoutCsrIsRebuiltCorrectly) {
    // The cursor-restore step is the easiest thing here to get wrong, and a
    // wrong fanout only shows up as a missed wake. Check the offsets directly.
    //   0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    Graph g({{}, {0}, {0}, {1, 2}});
    ASSERT_EQ(g.sort(), TOPO_OK);
    EXPECT_EQ(g.fanout_offsets[0], 0u);
    EXPECT_EQ(g.fanout_offsets[1], 2u);  // task 0 has two consumers
    EXPECT_EQ(g.fanout_offsets[2], 3u);  // task 1 has one
    EXPECT_EQ(g.fanout_offsets[3], 4u);  // task 2 has one
    EXPECT_EQ(g.fanout_offsets[4], 4u);  // task 3 has none
}
