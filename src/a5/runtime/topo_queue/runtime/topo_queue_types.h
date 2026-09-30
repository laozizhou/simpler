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
 * topo_queue wire types — the GM image an AICPU topological sort publishes and
 * every AICore worker consumes.
 *
 * Execution model: the AICPU orders the already-built DAG once, then idles.
 * There is one queue for both core types; an AICore peeks the head and claims
 * it by CAS only when the head task matches its own type, idling otherwise. A
 * claimed task waits for every predecessor's completion counter to read DONE,
 * runs, and publishes its own counter. No dispatch register, no ready queue,
 * no reclaim.
 *
 * Deadlock freedom comes from the ordering plus the claim rule, not from a
 * detector; the full argument is topo_worker.h's header. In short: the head
 * advances only when a matching core claims it, and the lowest-indexed claimed
 * task always has every predecessor retired, so someone always runs.
 *
 * Every struct here is copied to the device as-is, so all of them are trivially
 * copyable, standard-layout, and free of host pointers. Intra-image references
 * are indices into the arrays this header declares.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Address-space qualifier for every pointer that names a byte of the image.
// The AICore compiler makes it part of the type: a plain `T *` is Local Memory
// there, so an unqualified parameter cannot receive a pointer into GM and the
// call does not compile. It is empty on the AICPU and in host tests, where the
// keyword does not exist, which is why the worker loop and its platform policy
// are written in terms of this macro and compile unchanged for all three.
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
#define TOPO_GM __gm__
#else
#define TOPO_GM
#endif

// A host build has no CCE keywords. Defining them away lets the wire types, the
// layout builder and the worker loop compile into a host test, which is what
// keeps their invariants under test rather than only under the AICore compiler.
#ifndef __aicore__
#define __aicore__
#endif

namespace simpler::topo_queue {

// One counter per task, each alone in its own cache line.
//
// Padding is mandatory, not a tuning choice. An AICore writes back whole cache
// lines, so two cores publishing neighbouring counters in the same line would
// each flush their own stale copy of the other's bytes and the last writer would
// win -- a completed task silently reverting to pending. The failure is
// invisible under sim, whose cache model is a no-op, and appears only on
// silicon. See docs/hardware/cache-coherency.md.
// 64 is a literal here because the platform exposes no cache-line constant to
// derive it from: the rest of the tree hardcodes the same number as
// `alignas(64)` / `__attribute__((aligned(64)))` on its own device-visible
// structs (Handshake and DeviceRuntimeLaunchDesc among them). If a platform
// constant ever appears, this should become an alias for it rather than a
// second source of truth.
inline constexpr uint32_t CACHE_LINE_BYTES = 64;

// Counter states. PENDING must be 0 so a zeroed image starts correct.
//
// 64-bit because that is the only width the AICore GM primitives carry:
// topo_gm_{query,store,fetch_add,compare_exchange} are built on ld_dev /
// st_dev / atomicAdd / atomicCAS, all of which are 64-bit. A 32-bit counter
// would have no atomic load or store on silicon.
enum TaskCounterState : uint64_t {
    TASK_PENDING = 0,
    TASK_DONE = 1,
    // A worker that fails publishes FAILED instead of DONE, so waiters stop
    // rather than spin on a counter that will never reach DONE.
    TASK_FAILED = 2,
};

struct alignas(CACHE_LINE_BYTES) TaskCounter {
    volatile uint64_t state;
    uint8_t pad[CACHE_LINE_BYTES - sizeof(uint64_t)];
};

static_assert(sizeof(TaskCounter) == CACHE_LINE_BYTES, "one counter per cache line");
static_assert(alignof(TaskCounter) == CACHE_LINE_BYTES, "counters must not share a line");

// Compressed sparse row over the fanin edges.
//
// Written once by the AICPU before any worker starts and read-only afterwards,
// so it needs no coherency protocol of its own -- one publish barrier covers
// the whole image.
//
//   predecessors of task t == fanin_ids[fanin_offsets[t] .. fanin_offsets[t+1])
//
// fanin_offsets therefore holds task_count + 1 entries.
struct FaninCsr {
    uint32_t task_count;
    uint32_t edge_count;
    uint32_t offsets_index;  // index into the image's uint32 pool
    uint32_t ids_index;
};

// Queue head, alone in its line: every worker peeks it and matching-type
// workers CAS it forward, so it must not share a line with anything a worker
// reads on the hot path. 64-bit for the same reason the counters are -- the
// device's atomicCAS carries no narrower width.
struct alignas(CACHE_LINE_BYTES) QueueHead {
    volatile uint64_t next;
    uint8_t pad[CACHE_LINE_BYTES - sizeof(uint64_t)];
};

static_assert(sizeof(QueueHead) == CACHE_LINE_BYTES, "queue head owns its line");

// Run-wide control. Each field sits in its own line rather than sharing one:
// `error` is read by every spinning worker on every pass, while `retired` is
// written by each worker as it leaves. Packed together, the completion writes
// would invalidate the line the whole device is polling.
struct alignas(CACHE_LINE_BYTES) RunControl {
    volatile uint64_t error;  // 0 = healthy, else the latched error code
    uint8_t pad0[CACHE_LINE_BYTES - sizeof(uint64_t)];
    volatile uint64_t retired;  // summed once per core as it leaves, not per task
    uint8_t pad1[CACHE_LINE_BYTES - sizeof(uint64_t)];
};

static_assert(sizeof(RunControl) == 2 * CACHE_LINE_BYTES, "each run-control field owns a line");
static_assert(offsetof(RunControl, retired) == CACHE_LINE_BYTES, "retired must not share error's line");

// What the claim gate needs to know about one task. Everything the task needs
// to RUN lives in its ready-made DispatchPayload in the image's payload region,
// indexed by task_id; this entry exists so the gate can read a task's type
// without touching that (much larger) record.
struct TaskEntry {
    uint32_t task_id;       // index into the counter array, the CSR and the payload region
    uint8_t core_type;      // CORE_TYPE_AIC or CORE_TYPE_AIV
    uint8_t reserved[3];
};

static_assert(std::is_trivially_copyable_v<TaskEntry> && std::is_standard_layout_v<TaskEntry>);
static_assert(std::is_trivially_copyable_v<TaskCounter> && std::is_standard_layout_v<TaskCounter>);
static_assert(std::is_trivially_copyable_v<QueueHead> && std::is_standard_layout_v<QueueHead>);
static_assert(std::is_trivially_copyable_v<RunControl> && std::is_standard_layout_v<RunControl>);

// Wire encoding of a core type, narrowed to the byte TaskEntry carries. These
// are constants rather than an enum because `CoreType` is already the platform's
// type (common/core_type.h) and must keep exactly one meaning; this header
// cannot include that one, since the algorithm layer builds host-side with no
// platform headers on the include path. aicore_executor.cpp sees both
// definitions and static_asserts that the values still agree.
inline constexpr uint8_t CORE_TYPE_AIC = 0;
inline constexpr uint8_t CORE_TYPE_AIV = 1;


// Error codes latched into RunControl::error.
enum TopoQueueError : uint32_t {
    TOPO_OK = 0,
    TOPO_ERR_TASK_FAILED = 1,        // a task reported failure
    TOPO_ERR_PREDECESSOR_FAILED = 2, // a predecessor was FAILED, so this task cannot run
    TOPO_ERR_WAIT_TIMEOUT = 3,       // a predecessor did not retire within the budget
    TOPO_ERR_CYCLE = 5,              // the topological sort could not order the graph
    TOPO_ERR_BAD_IMAGE = 6,          // image/executor version skew (payload stride mismatch)
};

}  // namespace simpler::topo_queue
