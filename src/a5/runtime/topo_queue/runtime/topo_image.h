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
 * The one GM object the AICPU publishes and every AICore reads.
 *
 * Layout rule: the header carries byte offsets from its own base, never
 * pointers, per the wire contract in .claude/rules/codestyle.md rule 8.
 *
 * One deliberate exception makes the FILLED image position-DEPENDENT: the
 * payload region. HBG's materialize writes absolute addresses into each
 * DispatchPayload -- among them self-pointers (args' context slots point back
 * into the payload's own local/global context). So the image must be
 * materialized IN PLACE at its final GM address; building it elsewhere and
 * memcpy'ing it over would leave every context slot pointing at the old
 * location. This constrains the onboard GM-allocation fix: allocate first,
 * fill second -- never fill-then-copy.
 *
 * Ordering rule: everything except the counters, the queue head and the run
 * control is written once before any core starts and never written again. That
 * is what lets the workers read the tables with no coherency protocol of their
 * own -- one observe pass at bring-up covers all of it. The mutable objects
 * each own a cache line for the opposite reason.
 *
 *   header
 *     +-- order[]            task_count  uint32   the queue, in topological order
 *     +-- fanin_offsets[]    count + 1   uint32   CSR row starts
 *     +-- fanin_ids[]        edge_count  uint32   flattened predecessor ids
 *     +-- entries[]          task_count  TaskEntry
 *     +-- payloads[]         task_count  x payload_stride bytes   (64-aligned)
 *     +-- counters[]         task_count  TaskCounter   (cache-line aligned)
 *     +-- queue_head         1           QueueHead     (cache-line aligned)
 *     +-- run_control        1           RunControl    (two lines)
 *
 * The payloads region holds one ready-made DispatchPayload per task, indexed by
 * task_id, built by the AICPU with HBG's own materialize routine so the args
 * and context-slot conventions the kernels expect are never re-derived here.
 * This header deliberately does not include dispatch_payload.h: the region is
 * bytes plus a stride, and the two sides that agree on the element type
 * (topo_prepare.h writes it, the executor reads it) each static_assert the
 * stride against sizeof(DispatchPayload) where that type is visible.
 */

#include <cstdint>

#include "topo_queue_types.h"

namespace simpler::topo_queue {

inline constexpr uint32_t TOPO_IMAGE_MAGIC = 0x54505149u;  // "TPQI"
inline constexpr uint32_t TOPO_IMAGE_VERSION = 3;          // v3: scheduler trace-cell base

/*
 * Offsets are from the header's own base and must each land on a
 * CACHE_LINE_BYTES boundary for the mutable arrays and the payloads region.
 * The builder asserts that; a reader takes it on trust because it cannot
 * afford the check per task.
 */
// alignas(CACHE_LINE_BYTES), not alignas(8), and the difference is load-bearing.
//
// Every "owns its cache line" guarantee in this runtime is expressed as an
// offset from this header's base, so it is only a guarantee if the base itself
// sits on a line. An 8-byte-aligned image would leave each TaskCounter
// straddling two lines and sharing each with a neighbour -- exactly the
// last-writer-wins clobber the padding exists to prevent, and invisible under
// sim. Putting the requirement in the type means the allocator cannot get it
// wrong, and it also keeps sizeof a multiple of alignof, so the arrays that
// follow stay aligned however many offset fields are added later.
struct alignas(CACHE_LINE_BYTES) TopoImageHeader {
    uint32_t magic;
    uint32_t version;

    uint32_t task_count;  // total; the counter array's length
    uint32_t edge_count;

    /*
     * One queue for both core types, AIC and AIV tasks interleaved in
     * topological order. Every a5 core enters the same executor; a core whose
     * type does not match the head task leaves the head alone and idles --
     * the claim is a CAS, never a blind increment, so an index is only ever
     * consumed by a core that will run it. The full claim rule and the
     * deadlock-freedom argument live in topo_worker.h.
     */
    uint32_t order_offset;
    uint32_t queue_head_offset;

    // Byte offsets from this header's base.
    uint32_t fanin_offsets_offset;
    uint32_t fanin_ids_offset;
    uint32_t entries_offset;
    uint32_t payloads_offset;
    uint32_t payload_stride;  // bytes per task; writer and reader both pin it to sizeof(DispatchPayload)
    uint32_t counters_offset;
    uint32_t run_control_offset;

    uint32_t total_bytes;
    uint32_t reserved;

    /*
     * Absolute device address of this run's SchedulerTaskTrace array, or 0 when
     * the run captures no swimlane.
     *
     * An address rather than an offset because the array is not in this image:
     * it is the one HBG's host already reserves inside the resident scheduler
     * state, sized by the same task count, and whose device address that host
     * keeps so it can copy the cells back after the run. Writing the cells there
     * is what lets the run's per-task timing reach chip_swimlane_records.json
     * through the publication path HBG already owns.
     */
    uint64_t trace_cells_address;
};

static_assert(sizeof(TopoImageHeader) % 8 == 0, "header keeps the arrays that follow 8-byte aligned");
static_assert(
    alignof(TopoImageHeader) >= CACHE_LINE_BYTES,
    "the image base must sit on a cache line or every per-line guarantee below is void"
);
// The header is wider than one line, so a reader must invalidate it by size.
// Asserting the span here is what keeps a future offset field from pushing a
// live field into a line nobody invalidated.
inline constexpr uint32_t TOPO_HEADER_LINES = sizeof(TopoImageHeader) / CACHE_LINE_BYTES;
static_assert(sizeof(TopoImageHeader) % CACHE_LINE_BYTES == 0, "header spans whole lines");

/*
 * A reader's resolved view. Built once per core at bring-up so the hot loop
 * indexes arrays instead of re-deriving addresses from offsets.
 */
struct TopoImageView {
    TOPO_GM TopoImageHeader *header;
    TOPO_GM const uint32_t *order;
    TOPO_GM QueueHead *queue_head;
    TOPO_GM const uint32_t *fanin_offsets;
    TOPO_GM const uint32_t *fanin_ids;
    TOPO_GM const TaskEntry *entries;
    TOPO_GM uint8_t *payloads;  // task_count x payload_stride; element type agreed outside this header
    uint32_t payload_stride;
    TOPO_GM TaskCounter *counters;
    TOPO_GM RunControl *run_control;
};

namespace detail {

template <typename T>
inline __aicore__ TOPO_GM T *at(TOPO_GM TopoImageHeader *base, uint32_t offset) {
    return reinterpret_cast<TOPO_GM T *>(reinterpret_cast<TOPO_GM uint8_t *>(base) + offset);
}

}  // namespace detail

inline __aicore__ void topo_image_bind(TOPO_GM TopoImageHeader *header, TopoImageView &out) {
    out.header = header;
    out.order = detail::at<const uint32_t>(header, header->order_offset);
    out.queue_head = detail::at<QueueHead>(header, header->queue_head_offset);
    out.fanin_offsets = detail::at<const uint32_t>(header, header->fanin_offsets_offset);
    out.fanin_ids = detail::at<const uint32_t>(header, header->fanin_ids_offset);
    out.entries = detail::at<const TaskEntry>(header, header->entries_offset);
    out.payloads = detail::at<uint8_t>(header, header->payloads_offset);
    out.payload_stride = header->payload_stride;
    out.counters = detail::at<TaskCounter>(header, header->counters_offset);
    out.run_control = detail::at<RunControl>(header, header->run_control_offset);
}

/*
 * The task's ready-made dispatch payload, as raw bytes. The caller casts to the
 * DispatchPayload it shares with the writer and static_asserts the stride; this
 * accessor cannot, because keeping the image format free of runtime types is
 * what lets a host test hold the layout without the HBG include contract.
 */
inline __aicore__ TOPO_GM uint8_t *topo_image_payload(const TopoImageView &image, uint32_t task_id) {
    return image.payloads + static_cast<uint64_t>(task_id) * image.payload_stride;
}

/*
 * Size an image and fill in its offsets. The AICPU builder and every reader
 * share this so the two sides cannot disagree about the layout -- a builder
 * that computed offsets separately from the reader would be one
 * silent-corruption bug waiting for a field to be added.
 *
 * Returns the total byte size, or 0 when the shape does not fit the uint32
 * offsets the header carries (or the stride is unusable). Zero means "refuse
 * to build this image": the cursor is accumulated in 64 bits precisely so the
 * overflow is detected here rather than wrapping into a plausible-looking
 * small size, which both sides would then agree on while the arrays overlapped
 * each other.
 *
 * On success the caller allocates that much and memsets it to zero before
 * filling. Zeroing is what makes every counter start PENDING and the queue head
 * start at 0, so a run needs no separate reset pass.
 */
inline uint32_t topo_image_layout(
    uint32_t task_count, uint32_t edge_count, uint32_t payload_stride, TopoImageHeader &header
) {
    auto align_up = [](uint64_t v, uint64_t a) { return (v + a - 1u) / a * a; };

    // DispatchPayload is alignas(64); a stride that is not a whole number of
    // lines would misalign every element after the first.
    if (payload_stride == 0 || payload_stride % CACHE_LINE_BYTES != 0) {
        header.magic = 0;
        header.total_bytes = 0;
        return 0;
    }

    header.magic = TOPO_IMAGE_MAGIC;
    header.version = TOPO_IMAGE_VERSION;
    header.task_count = task_count;
    header.edge_count = edge_count;
    header.payload_stride = payload_stride;
    header.reserved = 0;
    header.trace_cells_address = 0;

    uint64_t cursor = align_up(sizeof(TopoImageHeader), 8u);

    header.order_offset = static_cast<uint32_t>(cursor);
    cursor += static_cast<uint64_t>(task_count) * sizeof(uint32_t);

    header.fanin_offsets_offset = static_cast<uint32_t>(cursor);
    cursor += (static_cast<uint64_t>(task_count) + 1u) * sizeof(uint32_t);

    header.fanin_ids_offset = static_cast<uint32_t>(cursor);
    cursor += static_cast<uint64_t>(edge_count) * sizeof(uint32_t);

    cursor = align_up(cursor, 8u);
    header.entries_offset = static_cast<uint32_t>(cursor);
    cursor += static_cast<uint64_t>(task_count) * sizeof(TaskEntry);

    cursor = align_up(cursor, CACHE_LINE_BYTES);
    header.payloads_offset = static_cast<uint32_t>(cursor);
    cursor += static_cast<uint64_t>(task_count) * payload_stride;

    // The mutable objects start on their own cache lines. Everything above is
    // read-only after bring-up, so it may share lines freely.
    cursor = align_up(cursor, CACHE_LINE_BYTES);
    header.counters_offset = static_cast<uint32_t>(cursor);
    cursor += static_cast<uint64_t>(task_count) * sizeof(TaskCounter);

    header.queue_head_offset = static_cast<uint32_t>(cursor);  // aligned: TaskCounter is a line
    cursor += sizeof(QueueHead);

    header.run_control_offset = static_cast<uint32_t>(cursor);
    cursor += sizeof(RunControl);

    if (cursor > 0xFFFFFFFFull) {
        // Leave nothing a caller could mistake for a usable image.
        header.magic = 0;
        header.total_bytes = 0;
        return 0;
    }

    header.total_bytes = static_cast<uint32_t>(cursor);
    return static_cast<uint32_t>(cursor);
}

}  // namespace simpler::topo_queue
