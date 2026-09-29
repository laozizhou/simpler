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
 * The AICPU's image builder, driven against a hand-built HBG graph.
 *
 * topo_prepare.h is the one piece of the AICPU that decides anything; the
 * executor around it is handshake plumbing. Faking the graph here -- byte-level,
 * through scheduler_graph.h's own offset constants -- is what lets the whole
 * builder run as a host unit test, including the materialize call, before a
 * simulator ever launches. The fake is written strictly through the published
 * offsets, so if HBG moves a field, this test moves with it or fails loudly,
 * never silently diverges.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "dispatch_payload.h"
#include "scheduler/scheduler_graph.h"
#include "topo_image.h"
#include "topo_prepare.h"
#include "topo_queue_types.h"

using namespace simpler::topo_queue;

namespace {

// Where the fake plants each variable-length area inside a task's 192-byte
// payload region (stride 320, payload at +128). Deltas are self-relative, from
// the field that stores them.
constexpr uint32_t kFaninArrayAt = 24;   // up to 4 predecessor ids
constexpr uint32_t kScalarsAt = 40;      // one uint64 scalar
constexpr uint32_t kTensorAt = 64;       // one 128-byte hbg tensor record

struct FakeGraph {
    static constexpr uint32_t kMaxTasks = 16;

    explicit FakeGraph(uint32_t task_count) : count(task_count) {
        storage.reset(static_cast<uint8_t *>(::operator new[](count * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE,
                                                              std::align_val_t{CACHE_LINE_BYTES})));
        std::memset(storage.get(), 0, count * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
        callable_table.assign(1024, 0);
        for (uint32_t t = 0; t < count; ++t) {
            set_u64(descriptor(t) + SCHEDULER_GRAPH_TASK_ID_OFFSET, t);
            callable_table[100 + t] = 0xA000u + t;
            // Single vector kernel in slot 1 by default; per-test overrides below.
            set_kernel_slots(t, SCHEDULER_GRAPH_INVALID_KERNEL_ID, 100 + static_cast<int32_t>(t),
                             SCHEDULER_GRAPH_INVALID_KERNEL_ID);
            // One tensor and one scalar per task, planted inside this task's
            // own payload region.
            set_i32(payload(t) + TASKPAYLOAD_TENSOR_COUNT_OFFSET, 1);
            set_i32(payload(t) + TASKPAYLOAD_SCALAR_COUNT_OFFSET, 1);
            set_i32(payload(t) + TASKPAYLOAD_TENSORS_DELTA_OFFSET,
                    static_cast<int32_t>(kTensorAt - TASKPAYLOAD_TENSORS_DELTA_OFFSET));
            set_i32(payload(t) + TASKPAYLOAD_SCALARS_DELTA_OFFSET,
                    static_cast<int32_t>(kScalarsAt - TASKPAYLOAD_SCALARS_DELTA_OFFSET));
            set_u64(payload(t) + kScalarsAt, 0x5CA1A0ull + t);
        }
    }

    uint8_t *descriptor(uint32_t t) { return storage.get() + t * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE; }
    uint8_t *payload(uint32_t t) { return descriptor(t) + SCHEDULER_GRAPH_PAYLOAD_OFFSET; }

    void set_kernel_slots(uint32_t t, int32_t aic, int32_t aiv0, int32_t aiv1) {
        int32_t *slots = reinterpret_cast<int32_t *>(descriptor(t) + SCHEDULER_GRAPH_KERNEL_IDS_OFFSET);
        slots[0] = aic;
        slots[1] = aiv0;
        slots[2] = aiv1;
    }

    void set_fanin(uint32_t t, std::vector<int32_t> producers) {
        set_i32(payload(t) + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET, static_cast<int32_t>(producers.size()));
        set_i32(payload(t) + TASKPAYLOAD_FANIN_DELTA_OFFSET,
                producers.empty() ? 0 : static_cast<int32_t>(kFaninArrayAt - TASKPAYLOAD_FANIN_DELTA_OFFSET));
        int32_t *ids = reinterpret_cast<int32_t *>(payload(t) + kFaninArrayAt);
        for (size_t k = 0; k < producers.size(); ++k) ids[k] = producers[k];
    }

    // kernel_id -> resolved entry, as the host's bind step would fill it.
    std::vector<uint64_t> callable_table;

    SchedulerGraphView view() {
        return SchedulerGraphView{reinterpret_cast<uint64_t>(storage.get()), 0, count, count - 1};
    }

    static void set_u64(uint8_t *at, uint64_t v) { std::memcpy(at, &v, sizeof(v)); }
    static void set_i32(uint8_t *at, int32_t v) { std::memcpy(at, &v, sizeof(v)); }

    struct Free {
        void operator()(uint8_t *p) const { ::operator delete[](p, std::align_val_t{CACHE_LINE_BYTES}); }
    };
    std::unique_ptr<uint8_t[], Free> storage;
    uint32_t count;
};

struct BuiltImage {
    std::unique_ptr<uint8_t[], FakeGraph::Free> bytes;
    TopoImageHeader *header = nullptr;
    TopoImageView view{};
    TopoPrepareResult result = TOPO_PREPARE_BAD_GRAPH;
    int64_t failed_task = -1;
};

BuiltImage build(FakeGraph &g) {
    BuiltImage out;
    const SchedulerGraphView graph = g.view();
    TopoImageHeader sizing{};
    const uint32_t bytes = topo_prepare_image_bytes(graph, sizing);
    if (bytes == 0) return out;
    out.bytes.reset(static_cast<uint8_t *>(::operator new[](bytes, std::align_val_t{CACHE_LINE_BYTES})));
    std::memset(out.bytes.get(), 0, bytes);
    out.header = reinterpret_cast<TopoImageHeader *>(out.bytes.get());
    *out.header = sizing;
    out.result = topo_prepare_fill(graph, g.callable_table.data(), g.callable_table.size(), out.header, &out.failed_task);
    topo_image_bind(out.header, out.view);
    return out;
}

}  // namespace

TEST(TopoPrepare, StridePinsToDispatchPayload)
{
    // Writer-side half of the two-sided pin; the executor asserts the same.
    FakeGraph g(1);
    g.set_fanin(0, {});
    BuiltImage img = build(g);
    ASSERT_EQ(img.result, TOPO_PREPARE_OK);
    EXPECT_EQ(img.header->payload_stride, sizeof(DispatchPayload));
    EXPECT_EQ(img.header->version, TOPO_IMAGE_VERSION);
}

TEST(TopoPrepare, BuildsIdentityOrderCsrAndTypedEntries)
{
    // A small diamond: 0 -> {1,2} -> 3, with 0,3 on AIC and 1,2 on AIV.
    FakeGraph g(4);
    g.set_kernel_slots(0, 900, SCHEDULER_GRAPH_INVALID_KERNEL_ID, SCHEDULER_GRAPH_INVALID_KERNEL_ID);
    g.callable_table[900] = 0xB900;
    g.set_kernel_slots(3, 903, SCHEDULER_GRAPH_INVALID_KERNEL_ID, SCHEDULER_GRAPH_INVALID_KERNEL_ID);
    g.callable_table[903] = 0xB903;
    g.set_fanin(0, {});
    g.set_fanin(1, {0});
    g.set_fanin(2, {0});
    g.set_fanin(3, {1, 2});

    BuiltImage img = build(g);
    ASSERT_EQ(img.result, TOPO_PREPARE_OK) << "failed at task " << img.failed_task;

    ASSERT_EQ(img.header->task_count, 4u);
    ASSERT_EQ(img.header->edge_count, 4u);
    for (uint32_t t = 0; t < 4; ++t) {
        EXPECT_EQ(img.view.order[t], t) << "v1 order is identity";
        EXPECT_EQ(img.view.entries[t].task_id, t);
    }
    EXPECT_EQ(img.view.entries[0].core_type, CORE_TYPE_AIC);
    EXPECT_EQ(img.view.entries[1].core_type, CORE_TYPE_AIV);
    EXPECT_EQ(img.view.entries[2].core_type, CORE_TYPE_AIV);
    EXPECT_EQ(img.view.entries[3].core_type, CORE_TYPE_AIC);

    // CSR rows exactly as the workers will read them.
    const uint32_t expect_offsets[5] = {0, 0, 1, 2, 4};
    for (uint32_t i = 0; i < 5; ++i) EXPECT_EQ(img.view.fanin_offsets[i], expect_offsets[i]);
    EXPECT_EQ(img.view.fanin_ids[0], 0u);
    EXPECT_EQ(img.view.fanin_ids[1], 0u);
    EXPECT_EQ(img.view.fanin_ids[2], 1u);
    EXPECT_EQ(img.view.fanin_ids[3], 2u);

    // Zeroed mutable tail == valid un-started run.
    EXPECT_EQ(img.view.queue_head->next, 0u);
    EXPECT_EQ(img.view.run_control->error, TOPO_OK);
    for (uint32_t t = 0; t < 4; ++t) EXPECT_EQ(img.view.counters[t].state, TASK_PENDING);
}

TEST(TopoPrepare, MaterializedPayloadCarriesHbgConventions)
{
    // The point of plan A: the payload's args must be exactly what HBG's own
    // materialize produces -- tensor ADDRESSES first, scalar VALUES after, and
    // the two context slots filled -- so the unified kernel signature works
    // unmodified. Checked against the fake's planted locations.
    FakeGraph g(2);
    g.set_fanin(0, {});
    g.set_fanin(1, {0});

    BuiltImage img = build(g);
    ASSERT_EQ(img.result, TOPO_PREPARE_OK);

    for (uint32_t t = 0; t < 2; ++t) {
        auto *payload = reinterpret_cast<DispatchPayload *>(topo_image_payload(img.view, t));
        EXPECT_EQ(payload->function_bin_addr, 0xA000ull + t) << "resolved address from descriptor offset 136";
        EXPECT_EQ(payload->args[0], reinterpret_cast<uint64_t>(g.payload(t) + kTensorAt))
            << "args[0] is the tensor's GM address inside HBG's task storage";
        EXPECT_EQ(payload->args[1], 0x5CA1A0ull + t) << "args[1] is the scalar's value";
        EXPECT_EQ(payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX], reinterpret_cast<uint64_t>(&payload->local_context));
        EXPECT_EQ(payload->args[PAYLOAD_GLOBAL_CONTEXT_INDEX], reinterpret_cast<uint64_t>(&payload->global_context));
        EXPECT_EQ(payload->local_context.block_idx, 0);
        EXPECT_EQ(payload->local_context.block_num, 1);
        EXPECT_EQ(payload->src_payload, 0u);
    }
}

TEST(TopoPrepare, RefusesAForwardFanin)
{
    // scheduler_graph.h pins producer < task_id; a forward edge would break the
    // identity-order proof, so the builder must refuse, not "fix".
    FakeGraph g(3);
    g.set_fanin(0, {});
    g.set_fanin(1, {2});  // forward reference
    g.set_fanin(2, {});

    BuiltImage img = build(g);
    EXPECT_EQ(img.result, TOPO_PREPARE_BAD_GRAPH);
    EXPECT_EQ(img.failed_task, 1);
}

TEST(TopoPrepare, RefusesAMultiKernelTask)
{
    // MIX shapes never reach this builder on the real path (the host's shape
    // gate selects legacy first), but the builder must not TRUST that: a task
    // with two live kernel slots has no single core type.
    FakeGraph g(2);
    g.set_fanin(0, {});
    g.set_fanin(1, {0});
    g.set_kernel_slots(1, 700, 701, SCHEDULER_GRAPH_INVALID_KERNEL_ID);  // AIC + AIV

    BuiltImage img = build(g);
    EXPECT_EQ(img.result, TOPO_PREPARE_BAD_GRAPH);
    EXPECT_EQ(img.failed_task, 1);
}

TEST(TopoPrepare, RefusesAnUnresolvedKernelAddress)
{
    FakeGraph g(1);
    g.set_fanin(0, {});
    g.callable_table[100] = 0;  // registration never resolved this kernel

    BuiltImage img = build(g);
    EXPECT_EQ(img.result, TOPO_PREPARE_BAD_CALLABLE);
    EXPECT_EQ(img.failed_task, 0);
}

TEST(TopoPrepare, SizingRefusesAGraphWithBrokenFaninCounts)
{
    FakeGraph g(2);
    g.set_fanin(0, {});
    g.set_fanin(1, {0});
    FakeGraph::set_i32(g.payload(1) + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET, -5);

    TopoImageHeader h{};
    EXPECT_EQ(topo_prepare_image_bytes(g.view(), h), 0u);
}
