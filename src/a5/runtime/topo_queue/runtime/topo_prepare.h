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
 * The AICPU's one job: turn HBG's uploaded graph into the topo_queue image.
 *
 *     read graph -> validate -> order -> fill tables -> done, get out of the way
 *
 * The skeleton follows clc_prepare.c from the author's esl_proxy prototype --
 * build an index, check the input, order, fill, verify, publish -- with two
 * substitutions. The input is HBG's graph already resident in GM (read through
 * scheduler_graph.h) instead of case data, and each task's arguments are not
 * re-derived: HBG's own scheduler_materialize_task_payload_resolved() builds a
 * ready-made DispatchPayload per task, so the args and context-slot conventions
 * the kernels expect are inherited, never re-implemented.
 *
 * Ordering: order[k] = k. HBG task ids are already a topological order -- the
 * graph validator rejects any fanin id >= its consumer's id (scheduler_graph.h
 * pins producer < task_id), so identity is correct by construction. It is still
 * VERIFIED here (order_is_topological over the CSR this builder itself wrote),
 * because the deadlock-freedom argument stands on this property and a check is
 * cheaper than the hang it prevents. The order[] array stays in the image even
 * though it is identity today: it is the seam where an est / level sort drops
 * in later without the device side changing at all.
 *
 * Everything in this header is plain logic over memory the caller hands in --
 * no handshake, no registers, no allocation -- precisely so the whole builder
 * runs under host unit tests. The AICPU executor wraps it with the handshake.
 */

#include <cstdint>

#include "dispatch_payload.h"
#include "scheduler/scheduler_graph.h"
#include "topo_image.h"
#include "topo_queue_types.h"
#include "topo_sort.h"

namespace simpler::topo_queue {

// The image's payload region and this builder must agree on the element; the
// reader (aicore executor) carries the same assert against the same type.
static_assert(sizeof(DispatchPayload) % CACHE_LINE_BYTES == 0, "payload stride must be whole cache lines");

enum TopoPrepareResult : uint32_t {
    TOPO_PREPARE_OK = 0,
    TOPO_PREPARE_BAD_GRAPH = 1,        // classify rejected a task (id echo, fanin range, dup, shape)
    TOPO_PREPARE_BAD_ORDER = 2,        // identity order failed the topological check
    TOPO_PREPARE_BAD_CALLABLE = 3,     // a task's resolved kernel address is 0
    TOPO_PREPARE_MATERIALIZE_FAILED = 4,
    TOPO_PREPARE_TOO_MANY_EDGES = 5,   // fanin total exceeded what the image was sized for
};

/*
 * Size the image for a graph. Walks the fanin counts once; the edge total is
 * what the CSR is sized by. Returns 0 on a graph the builder would refuse.
 */
inline uint32_t topo_prepare_image_bytes(const SchedulerGraphView &graph, TopoImageHeader &header) {
    if (graph.storage_address == 0 || graph.task_count == 0 || graph.task_count > 0x7FFFFFFFull) return 0;
    uint64_t edges = 0;
    for (int64_t t = 0; t < static_cast<int64_t>(graph.task_count); ++t) {
        __gm__ uint8_t *payload = scheduler_graph_payload(graph, t);
        const int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
        if (fanin_count < 0 || fanin_count > SCHEDULER_GRAPH_MAX_FANIN) return 0;
        edges += static_cast<uint64_t>(fanin_count);
    }
    if (edges > 0xFFFFFFFFull) return 0;
    return topo_image_layout(
        static_cast<uint32_t>(graph.task_count), static_cast<uint32_t>(edges),
        static_cast<uint32_t>(sizeof(DispatchPayload)), header
    );
}

/*
 * The subtask slot a single-kernel task occupies decides its core type: slot 0
 * is the cube kernel, slots 1 and 2 the vector kernels, mirroring the cluster's
 * 1 AIC + 2 AIV lanes. topo_queue only ever sees single-subtask tasks -- the
 * host runs the same shape gate before it selects this scheduling mode -- so
 * exactly one bit of active_mask is set.
 */
inline bool topo_prepare_core_type(const SchedulerTaskShape &shape, uint8_t *out_core_type) {
    switch (shape.active_mask) {
        case 1u: *out_core_type = CORE_TYPE_AIC; return true;
        case 2u:
        case 4u: *out_core_type = CORE_TYPE_AIV; return true;
        default: return false;  // zero or multiple subtasks: not a topo_queue shape
    }
}

inline int32_t topo_prepare_single_kernel_id(const SchedulerTaskShape &shape) {
    for (int32_t slot = 0; slot < 3; ++slot) {
        if (shape.kernel_ids[slot] != SCHEDULER_GRAPH_INVALID_KERNEL_ID) return shape.kernel_ids[slot];
    }
    return SCHEDULER_GRAPH_INVALID_KERNEL_ID;
}

/*
 * Fill a zeroed, laid-out image from the graph. On failure the image must be
 * treated as garbage; `out_failed_task` names the first task that broke, for
 * the log line the caller owes the user.
 *
 * The mutable tail (counters, head, run control) is already correct from the
 * caller's memset: PENDING == 0, head == 0, error == TOPO_OK. This function
 * writes only the read-only region.
 */
inline TopoPrepareResult topo_prepare_fill(
    const SchedulerGraphView &graph,
    __gm__ const uint64_t *callable_addresses,  // host's kernel_id -> resolved entry table
    uint64_t callable_count,
    TOPO_GM TopoImageHeader *header,
    int64_t *out_failed_task
) {
    TopoImageView image{};
    topo_image_bind(header, image);
    *out_failed_task = -1;

    TOPO_GM uint32_t *order = const_cast<TOPO_GM uint32_t *>(image.order);
    TOPO_GM uint32_t *fanin_offsets = const_cast<TOPO_GM uint32_t *>(image.fanin_offsets);
    TOPO_GM uint32_t *fanin_ids = const_cast<TOPO_GM uint32_t *>(image.fanin_ids);
    TOPO_GM TaskEntry *entries = const_cast<TOPO_GM TaskEntry *>(image.entries);

    const uint32_t task_count = header->task_count;
    uint32_t edge_cursor = 0;

    for (uint32_t t = 0; t < task_count; ++t) {
        *out_failed_task = static_cast<int64_t>(t);

        // classify re-checks everything this builder is about to trust: the id
        // echo, fanin range and duplicates, and that fanin ids are all smaller
        // than t -- which is exactly what makes identity order topological.
        SchedulerTaskShape shape{};
        const SchedulerGraphResult rc = scheduler_classify_task_shape(graph, static_cast<int64_t>(t), &shape);
        if (rc != SchedulerGraphResult::OK) return TOPO_PREPARE_BAD_GRAPH;

        uint8_t core_type = 0;
        if (!topo_prepare_core_type(shape, &core_type)) return TOPO_PREPARE_BAD_GRAPH;

        order[t] = t;
        entries[t].task_id = t;
        entries[t].core_type = core_type;
        entries[t].reserved[0] = entries[t].reserved[1] = entries[t].reserved[2] = 0;

        // Copy the fanin row into the image's CSR. The worker loop reads its
        // own flat arrays rather than chasing HBG's per-payload delta pointers
        // on every wait pass; the copy is one pass here, at bring-up.
        __gm__ uint8_t *payload = scheduler_graph_payload(graph, t);
        const int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
        fanin_offsets[t] = edge_cursor;
        if (edge_cursor + static_cast<uint32_t>(fanin_count) > header->edge_count) {
            return TOPO_PREPARE_TOO_MANY_EDGES;  // graph changed between sizing and filling
        }
        for (int32_t k = 0; k < fanin_count; ++k) {
            fanin_ids[edge_cursor++] = static_cast<uint32_t>(scheduler_graph_fanin_id(graph, t, k));
        }

        // The host resolved every kernel entry into the callable table the
        // resident scheduler also reads (addresses[kernel_id]); a miss or a
        // zero is a bind failure that must not reach a core as a callable.
        const int32_t kernel_id = topo_prepare_single_kernel_id(shape);
        if (kernel_id < 0 || static_cast<uint64_t>(kernel_id) >= callable_count) return TOPO_PREPARE_BAD_CALLABLE;
        const uint64_t resolved_addr = callable_addresses[kernel_id];
        if (resolved_addr == 0) return TOPO_PREPARE_BAD_CALLABLE;

        // HBG's own materialize fills function_bin_addr, args[] (tensor
        // addresses then scalars) and the local/global context slots the
        // unified kernel signature expects. Reusing it is the whole point of
        // storing ready-made payloads: the conventions are inherited, not
        // re-implemented, so they cannot drift.
        SchedulerTaskInfo info{static_cast<int64_t>(t), kernel_id,
                               /*subtask_slot=*/0, static_cast<CoreType>(core_type)};
        __gm__ DispatchPayload *slot = reinterpret_cast<__gm__ DispatchPayload *>(topo_image_payload(image, t));
        if (scheduler_materialize_task_payload_resolved(graph, info, resolved_addr, slot) !=
            SchedulerGraphResult::OK) {
            return TOPO_PREPARE_MATERIALIZE_FAILED;
        }
    }
    fanin_offsets[task_count] = edge_cursor;
    *out_failed_task = -1;

    // The proof in topo_worker.h stands on this property; verifying it here
    // costs O(V+E) once, against the CSR exactly as the workers will read it.
    // scratch: reuse the order array? No -- it is live. The caller passes none,
    // so verify with the entries' reserved space is impossible too; instead the
    // check below recomputes positions inline (identity order makes it trivial:
    // position[id] == id, so the condition is fanin_ids[e] < t for every row).
    for (uint32_t t = 0; t < task_count; ++t) {
        for (uint32_t e = fanin_offsets[t]; e < fanin_offsets[t + 1]; ++e) {
            if (fanin_ids[e] >= t) {
                *out_failed_task = static_cast<int64_t>(t);
                return TOPO_PREPARE_BAD_ORDER;
            }
        }
    }
    return TOPO_PREPARE_OK;
}

}  // namespace simpler::topo_queue
