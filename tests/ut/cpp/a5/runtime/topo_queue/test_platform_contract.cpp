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
 * Holds the Platform policy contract to one shape.
 *
 * topo_worker.h is a template, so a policy with a wrong signature fails only
 * when something instantiates it -- and the device policy is instantiated by
 * the AICore build, which a host test never runs. A mismatch would therefore
 * surface as a ccec error on silicon, long after the host tests went green.
 *
 * These checks instantiate the loop against a policy whose members are declared
 * exactly as the contract states, so the contract is compiled here rather than
 * only described in a comment. The device policy itself cannot be included --
 * it needs __aicore__ and the CCE intrinsics -- so what this pins down is the
 * signature every conforming policy must carry, which the device one is then
 * written against.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "topo_queue_types.h"
#include "topo_worker.h"

using namespace simpler::topo_queue;

namespace {

// The contract, spelled out. Every method matches topo_worker.h's documented
// signature exactly; changing one here without changing the header makes the
// instantiation below fail.
//
// The queue head is served from the policy rather than from the struct, because
// that is what the contract says: the loop never reads QueueHead::next itself,
// it only asks the policy for the next index. A policy that ignored the struct
// entirely would still be conforming.
struct ContractPolicy {
    uint64_t next_index = 0;

    uint64_t load_queue_head(const QueueHead *) { return next_index; }
    bool try_claim_queue_head(QueueHead *, uint64_t expected) {
        if (expected != next_index) return false;
        ++next_index;
        return true;
    }
    uint64_t load_counter_acquire(const TaskCounter *) { return TASK_DONE; }
    void store_counter_release(TaskCounter *, uint64_t) {}
    uint64_t load_error_relaxed(const RunControl *) { return TOPO_OK; }
    void latch_error(RunControl *, uint64_t) {}
    uint64_t now_ticks() { return 0; }
    uint64_t timeout_ticks() { return 1; }
    void spin_hint() {}
};

}  // namespace

TEST(PlatformContract, LoopInstantiatesAgainstTheDocumentedSignature) {
    ContractPolicy plat;
    QueueHead head{};
    RunControl control{};
    uint32_t order[1] = {0};
    TaskCounter counters[1] = {};
    uint32_t fanin_offsets[2] = {0, 0};
    uint32_t fanin_ids[1] = {0};
    TaskEntry entries[1] = {};

    // The policy hands out index 0 first, so the one task runs; the second call
    // hands out 1, which is past the end and drains. Both branches of the step
    // are therefore instantiated, which is the point of the case.
    const WorkerStep first = worker_step(
        plat, &head, &control, order, /*task_count=*/1, CORE_TYPE_AIC, counters, fanin_offsets, fanin_ids, entries,
        [](const TaskEntry &) { return true; }
    );
    EXPECT_EQ(first, WorkerStep::RAN_TASK);

    const WorkerStep second = worker_step(
        plat, &head, &control, order, /*task_count=*/1, CORE_TYPE_AIC, counters, fanin_offsets, fanin_ids, entries,
        [](const TaskEntry &) { return true; }
    );
    EXPECT_EQ(second, WorkerStep::QUEUE_DRAINED);
}

TEST(PlatformContract, WireTypesAreTheWidthTheDeviceAtomicsCarry) {
    // topo_gm_{fetch_add,query,store,compare_exchange} are 64-bit only:
    // they resolve to atomicAdd / ld_dev / st_dev / atomicCAS. A 32-bit field
    // here would have no atomic form on silicon, so the widths are part of the
    // wire contract rather than a choice.
    static_assert(std::is_same_v<decltype(QueueHead::next), volatile uint64_t>);
    static_assert(std::is_same_v<decltype(TaskCounter::state), volatile uint64_t>);
    static_assert(std::is_same_v<decltype(RunControl::error), volatile uint64_t>);
    static_assert(std::is_same_v<decltype(RunControl::retired), volatile uint64_t>);
    SUCCEED();
}

TEST(PlatformContract, EveryHotFieldOwnsItsCacheLine) {
    // The padding is the only defence against two AICores clobbering each
    // other's fields on write-back, and sim cannot reproduce that failure. The
    // layout is therefore asserted rather than tested.
    static_assert(sizeof(TaskCounter) == CACHE_LINE_BYTES);
    static_assert(alignof(TaskCounter) == CACHE_LINE_BYTES);
    static_assert(sizeof(QueueHead) == CACHE_LINE_BYTES);
    static_assert(alignof(QueueHead) == CACHE_LINE_BYTES);
    static_assert(sizeof(RunControl) == 2 * CACHE_LINE_BYTES);
    SUCCEED();
}

TEST(PlatformContract, WireStructsSurviveAMemcpyToTheDevice) {
    // Everything that crosses to GM is copied as bytes and read back in place,
    // so none of it may carry a vtable, a host pointer, or padding the two sides
    // disagree about.
    static_assert(std::is_trivially_copyable_v<TaskCounter>);
    static_assert(std::is_trivially_copyable_v<QueueHead>);
    static_assert(std::is_trivially_copyable_v<RunControl>);
    static_assert(std::is_trivially_copyable_v<TaskEntry>);
    static_assert(std::is_standard_layout_v<TaskCounter>);
    static_assert(std::is_standard_layout_v<QueueHead>);
    static_assert(std::is_standard_layout_v<RunControl>);
    static_assert(std::is_standard_layout_v<TaskEntry>);
    SUCCEED();
}
