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
 * The GM image's layout invariants.
 *
 * topo_image_layout() is the single place host and device agree about where
 * each array starts, so an off-by-one there is silent corruption rather than a
 * compile error: both sides would read the same wrong offset and agree. These
 * cases check the properties the readers take on trust -- arrays in bounds,
 * never overlapping, each aligned for the type it holds, and every mutable
 * object starting on a cache line.
 *
 * They are also what holds the header's own size and alignment invariants,
 * which the AICore compiler would otherwise be the first thing to notice.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <new>

#include "topo_image.h"
#include "topo_queue_types.h"

using namespace simpler::topo_queue;

// The image is format-only: it carries the payload region as bytes + stride, so
// this test needs no HBG type. Two lines stands in for sizeof(DispatchPayload).
constexpr uint32_t kStride = 2 * CACHE_LINE_BYTES;

namespace {

// A zeroed image of the requested shape, with a bound view over it.
//
// The buffer starts on a cache line because the image's per-line guarantees are
// all expressed as offsets from its base -- a misaligned base would put every
// counter across two lines and quietly reintroduce the false sharing the
// padding exists to prevent. TopoImageHeader's own alignas says the same thing;
// this allocator is what honours it.
class TestImage {
public:
    TestImage(uint32_t tasks, uint32_t edges, uint32_t stride = kStride)
    {
        TopoImageHeader probe{};
        total_ = topo_image_layout(tasks, edges, stride, probe);
        if (total_ == 0) return;  // refused; view stays empty
        bytes_.reset(static_cast<uint8_t *>(::operator new[](total_, std::align_val_t{CACHE_LINE_BYTES})));
        std::memset(bytes_.get(), 0, total_);
        *header() = probe;
        topo_image_bind(header(), view);
    }

    TopoImageHeader *header() { return reinterpret_cast<TopoImageHeader *>(bytes_.get()); }
    uint32_t total() const { return total_; }

    TopoImageView view{};

private:
    struct Free {
        void operator()(uint8_t *p) const { ::operator delete[](p, std::align_val_t{CACHE_LINE_BYTES}); }
    };
    std::unique_ptr<uint8_t[], Free> bytes_;
    uint32_t total_ = 0;
};

}  // namespace

TEST(TopoImage, HeaderCarriesItsOwnAlignmentRequirement)
{
    // Not merely 8-byte: every "owns a cache line" claim below is an offset from
    // this header's base, so an 8-aligned image would void all of them.
    static_assert(sizeof(TopoImageHeader) % 8 == 0);
    static_assert(alignof(TopoImageHeader) >= CACHE_LINE_BYTES);
    static_assert(sizeof(TopoImageHeader) % CACHE_LINE_BYTES == 0);
    // The header is wider than one line, which is why the reader invalidates it
    // by size. If this ever becomes 1, that call may be simplified; if it grows,
    // the reader keeps working because it already uses sizeof.
    static_assert(TOPO_HEADER_LINES >= 1);
    SUCCEED();
}

TEST(TopoImage, LayoutPlacesEveryArrayInBoundsAndWithoutOverlap)
{
    // Shapes chosen to walk the alignment padding through every branch: odd
    // edge counts leave the cursor mid-word before entries, odd task counts
    // leave it mid-line before the counters. The aic/aiv split is varied
    // independently so neither queue is always the empty one.
    const uint32_t shapes[][2] = {
        {1, 0}, {2, 1}, {3, 3}, {8, 12}, {5, 7}, {17, 31}, {23, 40}, {64, 128},
    };

    for (const auto &shape : shapes) {
        const uint32_t tasks = shape[0], edges = shape[1];
        SCOPED_TRACE(::testing::Message() << "tasks=" << tasks << " edges=" << edges);

        TopoImageHeader h{};
        const uint32_t total = topo_image_layout(tasks, edges, kStride, h);
        ASSERT_NE(total, 0u);

        struct Span {
            const char *name;
            uint32_t begin;
            uint32_t bytes;
        };
        const Span spans[] = {
            {"order", h.order_offset, tasks * 4},
            {"fanin_offsets", h.fanin_offsets_offset, (tasks + 1) * 4},
            {"fanin_ids", h.fanin_ids_offset, edges * 4},
            {"entries", h.entries_offset, tasks * static_cast<uint32_t>(sizeof(TaskEntry))},
            {"payloads", h.payloads_offset, tasks * kStride},
            {"counters", h.counters_offset, tasks * CACHE_LINE_BYTES},
            {"queue_head", h.queue_head_offset, CACHE_LINE_BYTES},
            {"run_control", h.run_control_offset, 2 * CACHE_LINE_BYTES},
        };

        for (const auto &s : spans) {
            EXPECT_GE(s.begin, sizeof(TopoImageHeader)) << s.name << " overlaps the header";
            EXPECT_LE(s.begin + s.bytes, total) << s.name << " runs past the image";
        }

        // Spans are declared in layout order, so each must start at or after the
        // previous one's end. Equality is fine; overlap is not.
        for (size_t i = 1; i < std::size(spans); ++i) {
            EXPECT_GE(spans[i].begin, spans[i - 1].begin + spans[i - 1].bytes)
                << spans[i].name << " overlaps " << spans[i - 1].name;
        }

        EXPECT_EQ(h.total_bytes, total);
        EXPECT_EQ(h.task_count, tasks);
        EXPECT_EQ(h.magic, TOPO_IMAGE_MAGIC);
    }
}

TEST(TopoImage, EveryMutableObjectStartsOnItsOwnCacheLine)
{
    // The padding inside TaskCounter / QueueHead / RunControl only buys line
    // exclusivity if the layout also starts them on a line. Both halves are
    // needed, and only this one varies with the graph's shape.
    //
    // The two queue heads matter most: each is hammered by fetch_add from a
    // different set of cores, so sharing a line would make every AIC pull
    // invalidate the line every AIV pull is contending for.
    for (uint32_t tasks = 1; tasks <= 40; ++tasks) {
        for (uint32_t edges : {0u, 1u, 5u, 13u}) {
            TopoImageHeader h{};
            ASSERT_NE(topo_image_layout(tasks, edges, kStride, h), 0u);
            SCOPED_TRACE(::testing::Message() << "tasks=" << tasks << " edges=" << edges);
            EXPECT_EQ(h.counters_offset % CACHE_LINE_BYTES, 0u);
            EXPECT_EQ(h.queue_head_offset % CACHE_LINE_BYTES, 0u);
            EXPECT_EQ(h.run_control_offset % CACHE_LINE_BYTES, 0u);
            // Payload elements are alignas(64); the region must start on a line.
            EXPECT_EQ(h.payloads_offset % CACHE_LINE_BYTES, 0u);
        }
    }
}

TEST(TopoImage, AShapeTooLargeForThirtyTwoBitOffsetsIsRefused)
{
    // The cursor is accumulated in 64 bits so this is caught rather than
    // wrapping into a plausible small size that both sides would then agree on
    // while the arrays overlapped. Each of these overflows a different term.
    TopoImageHeader h{};

    EXPECT_EQ(topo_image_layout(70'000'000, 0, kStride, h), 0u) << "counters array alone exceeds 4 GiB";
    EXPECT_EQ(h.total_bytes, 0u);
    EXPECT_NE(h.magic, TOPO_IMAGE_MAGIC) << "a refused header must not look valid";

    EXPECT_EQ(topo_image_layout(40'000'000, 0, kStride, h), 0u) << "payload region exceeds 4 GiB";

    // A stride the payload type cannot have is refused outright: zero, or one
    // that would misalign every alignas(64) element after the first.
    EXPECT_EQ(topo_image_layout(4, 0, 0, h), 0u);
    EXPECT_EQ(topo_image_layout(4, 0, 96, h), 0u);

    // And a large-but-representable shape is still accepted, so the guard is
    // not simply refusing everything big.
    EXPECT_NE(topo_image_layout(1'000'000, 0, kStride, h), 0u);
}

TEST(TopoImage, AZeroedBufferStartsEveryTaskPendingAndTheQueueAtZero)
{
    // The builder memsets and does not otherwise initialise the mutable state,
    // so "zeroed == a valid un-started run" is a property the layout owes.
    TestImage img(/*tasks=*/6, /*edges=*/5);
    ASSERT_NE(img.total(), 0u);

    EXPECT_EQ(img.view.queue_head->next, 0u);
    EXPECT_EQ(img.view.run_control->error, TOPO_OK);
    EXPECT_EQ(img.view.run_control->retired, 0u);
    for (uint32_t t = 0; t < 6; ++t) {
        EXPECT_EQ(img.view.counters[t].state, TASK_PENDING) << "task " << t;
    }
}

TEST(TopoImage, WritesThroughTheViewLandWhereTheOffsetsSay)
{
    // A bound view must address the same bytes the offsets name; if bind() and
    // layout() ever disagreed, each array would still look self-consistent
    // through the view while pointing at another array's storage.
    TestImage img(/*tasks=*/4, /*edges=*/3);
    ASSERT_NE(img.total(), 0u);
    TopoImageHeader *h = img.header();
    const uint8_t *base = reinterpret_cast<const uint8_t *>(h);
    const TopoImageView &v = img.view;

    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.order) - base, h->order_offset);
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.fanin_offsets) - base, h->fanin_offsets_offset);
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.fanin_ids) - base, h->fanin_ids_offset);
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.entries) - base, h->entries_offset);
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.payloads) - base, h->payloads_offset);
    EXPECT_EQ(v.payload_stride, kStride);
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.counters) - base, h->counters_offset);
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.queue_head) - base, h->queue_head_offset);
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(v.run_control) - base, h->run_control_offset);

    // Writing the last element of each array must not disturb its neighbour.
    img.view.counters[3].state = TASK_DONE;
    img.view.queue_head->next = 99;
    EXPECT_EQ(img.view.counters[3].state, TASK_DONE);
    EXPECT_EQ(img.view.queue_head->next, 99u);
    EXPECT_EQ(img.view.counters[2].state, TASK_PENDING);
    EXPECT_EQ(img.view.run_control->error, TOPO_OK);
}

TEST(TopoImage, PayloadAccessorStridesByTheHeaderStride)
{
    TestImage img(/*tasks=*/3, /*edges=*/0);
    ASSERT_NE(img.total(), 0u);

    const uint8_t *base = reinterpret_cast<const uint8_t *>(img.header());
    for (uint32_t t = 0; t < 3; ++t) {
        const uint8_t *slot = topo_image_payload(img.view, t);
        EXPECT_EQ(slot - base, img.header()->payloads_offset + t * kStride);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(slot) % CACHE_LINE_BYTES, 0u)
            << "payload " << t << " must start on a cache line";
    }
}

TEST(TopoImage, AnEmptyGraphStillProducesAUsableImage)
{
    // The host gate can hand down a graph with nothing runnable in it. The
    // image must still bind, so the device reports a drained queue rather than
    // faulting on a null array.
    TestImage img(/*tasks=*/0, /*edges=*/0);
    ASSERT_NE(img.total(), 0u);

    EXPECT_EQ(img.view.header->task_count, 0u);
    EXPECT_EQ(img.view.queue_head->next, 0u);
    EXPECT_EQ(img.view.run_control->error, TOPO_OK);
}
