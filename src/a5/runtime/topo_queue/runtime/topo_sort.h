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
 * Kahn topological sort over a fanin CSR, plus the fanout CSR it builds on the
 * way. The AICPU runs this once per run; nothing calls it again.
 *
 * The queue order this produces is what makes the worker loop deadlock-free, so
 * the ordering property is an invariant of the whole runtime, not a detail of
 * this file: a predecessor always lands at a lower queue index than its
 * consumers. topo_worker.h states the argument that rests on it.
 *
 * Caller-provided storage only -- no allocation. The AICPU runs with a fixed
 * arena and must not fail on a heap it does not have.
 */

#include <cstdint>

#include "topo_queue_types.h"

namespace simpler::topo_queue {

/*
 * Scratch the sort needs beyond its output. All of it is caller-owned and sized
 * from task_count / edge_count, so a caller can place it in the arena.
 *
 *   in_degree        task_count entries
 *   fanout_offsets   task_count + 1 entries
 *   fanout_ids       edge_count entries
 *   frontier         task_count entries (ring of ready-but-unemitted tasks)
 */
struct TopoSortScratch {
    uint32_t *in_degree;
    uint32_t *fanout_offsets;
    uint32_t *fanout_ids;
    uint32_t *frontier;
};

/*
 * Order `task_count` tasks so that every fanin edge points backwards.
 *
 * Inputs are the fanin CSR the host produced: predecessors of task t are
 * fanin_ids[fanin_offsets[t] .. fanin_offsets[t + 1]).
 *
 * `out_order` receives task ids in execution order and must hold task_count
 * entries. Returns TOPO_OK, or TOPO_ERR_CYCLE when the graph does not order --
 * which also covers a malformed CSR, since an out-of-range predecessor leaves
 * its consumer's in-degree unsatisfiable.
 *
 * Kahn rather than DFS: the frontier is explicit, so this runs in bounded stack
 * on a device thread, and the "emitted < task_count" test at the end detects a
 * cycle without a second traversal.
 */
inline uint32_t topo_sort(
    uint32_t task_count,
    uint32_t edge_count,
    const uint32_t *fanin_offsets,
    const uint32_t *fanin_ids,
    const TopoSortScratch &scratch,
    uint32_t *out_order
) {
    if (task_count == 0) return TOPO_OK;

    uint32_t *in_degree = scratch.in_degree;
    uint32_t *fanout_offsets = scratch.fanout_offsets;
    uint32_t *fanout_ids = scratch.fanout_ids;
    uint32_t *frontier = scratch.frontier;

    // Pass 1: in-degree per task, and the fanout bucket size per producer.
    // fanout_offsets is used as a counter array first and turned into offsets in
    // pass 2, so it carries task_count + 1 entries throughout.
    for (uint32_t t = 0; t <= task_count; ++t) fanout_offsets[t] = 0;
    for (uint32_t t = 0; t < task_count; ++t) in_degree[t] = 0;

    for (uint32_t t = 0; t < task_count; ++t) {
        const uint32_t begin = fanin_offsets[t];
        const uint32_t end = fanin_offsets[t + 1];
        if (begin > end || end > edge_count) return TOPO_ERR_CYCLE;
        for (uint32_t e = begin; e < end; ++e) {
            const uint32_t producer = fanin_ids[e];
            // An out-of-range producer would corrupt the fanout arrays; refusing
            // here keeps the sort total on any input the host hands over.
            if (producer >= task_count) return TOPO_ERR_CYCLE;
            in_degree[t]++;
            fanout_offsets[producer + 1]++;
        }
    }

    // Pass 2: prefix-sum the bucket sizes into offsets.
    for (uint32_t t = 0; t < task_count; ++t) {
        fanout_offsets[t + 1] += fanout_offsets[t];
    }

    // Pass 3: fill the fanout ids. `cursor` walks each producer's bucket; it
    // reuses in_degree's storage is NOT possible (still needed), so the fill
    // uses fanout_offsets[producer] as a moving cursor and restores it after.
    // Restoring by a backwards shift avoids a second scratch array.
    for (uint32_t t = 0; t < task_count; ++t) {
        for (uint32_t e = fanin_offsets[t]; e < fanin_offsets[t + 1]; ++e) {
            const uint32_t producer = fanin_ids[e];
            fanout_ids[fanout_offsets[producer]++] = t;
        }
    }
    // fanout_offsets[p] now holds the END of p's bucket, i.e. the original
    // fanout_offsets[p + 1]. Shift back so offsets[p] is p's start again.
    for (uint32_t t = task_count; t > 0; --t) {
        fanout_offsets[t] = fanout_offsets[t - 1];
    }
    fanout_offsets[0] = 0;

    // Pass 4: Kahn. The frontier is a FIFO over the scratch array; head and tail
    // never wrap because each task enters at most once.
    uint32_t head = 0;
    uint32_t tail = 0;
    for (uint32_t t = 0; t < task_count; ++t) {
        if (in_degree[t] == 0) frontier[tail++] = t;
    }

    uint32_t emitted = 0;
    while (head < tail) {
        const uint32_t t = frontier[head++];
        out_order[emitted++] = t;
        for (uint32_t e = fanout_offsets[t]; e < fanout_offsets[t + 1]; ++e) {
            const uint32_t consumer = fanout_ids[e];
            if (--in_degree[consumer] == 0) frontier[tail++] = consumer;
        }
    }

    // A graph that does not drain has a cycle: the tasks left behind all still
    // owe an in-degree that nothing can retire.
    return emitted == task_count ? TOPO_OK : TOPO_ERR_CYCLE;
}

/*
 * Check that `order` really is a topological order of the fanin CSR -- every
 * predecessor appears at a lower position. Used by the unit tests. The shipped
 * builder (topo_prepare.h) uses identity order and carries its own inline
 * equivalent of this check; this standalone form exists so tests can verify
 * arbitrary orders. The worker loop's deadlock-freedom argument rests on
 * exactly this property, so it is worth being able to assert it directly.
 */
inline bool order_is_topological(
    uint32_t task_count, const uint32_t *fanin_offsets, const uint32_t *fanin_ids, const uint32_t *order,
    uint32_t *position_scratch
) {
    for (uint32_t i = 0; i < task_count; ++i) position_scratch[order[i]] = i;
    for (uint32_t i = 0; i < task_count; ++i) {
        const uint32_t t = order[i];
        for (uint32_t e = fanin_offsets[t]; e < fanin_offsets[t + 1]; ++e) {
            if (position_scratch[fanin_ids[e]] >= i) return false;
        }
    }
    return true;
}

}  // namespace simpler::topo_queue
