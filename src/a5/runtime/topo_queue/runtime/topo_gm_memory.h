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
 * The GM primitives topo_queue's AICore side is built from.
 *
 * Every one of these is a thin wrapper that resolves the silicon/sim split:
 * ld_dev, st_dev, atomicAdd and atomicCAS under __CCE_AICORE__, and the
 * compiler's __atomic_* builtins otherwise. They exist here, rather than being
 * taken from a5 HBG's scheduler_memory.h, for two reasons.
 *
 * The first is coupling. This runtime deliberately shares no data model with a5
 * HBG's scheduler, which is under active development; borrowing its memory
 * header would have reintroduced exactly that dependency through the back door.
 *
 * The second is a concrete defect that dependency caused. scheduler_memory.h
 * pulls in scheduler_types.h, which pulls common/host_build_graph/runtime_types.h,
 * which contains:
 *
 *     #if __has_include("spin_hint.h")
 *     #include "spin_hint.h"
 *     #else
 *     #define SPIN_WAIT_HINT() ((void)0)
 *     #endif
 *
 * spin_hint.h lives under platform/{sim,onboard}/aicpu/ and is therefore NOT on
 * the AICore target's include path. The #else branch wins and redefines the
 * platform's SPIN_WAIT_HINT to a no-op for the whole translation unit -- so
 * under simulation, where every AICore is a host thread, a spin loop would
 * never yield. That is the starvation mode sim's own SPIN_WAIT_HINT exists to
 * prevent. Depending only on the platform layer keeps the real macro.
 *
 * All widths are 64-bit, which is why the wire types are: the device carries no
 * 32-bit atomic load or store.
 */

#include <cstdint>

#include "aicore/aicore.h"
#include "topo_queue_types.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

namespace simpler::topo_queue {

/*
 * Full barrier. Orders everything before it against everything after, which is
 * what both the publish and the consume side of the counter protocol need.
 */
inline __aicore__ void topo_full_barrier() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/*
 * Make a peer's write to this line visible. ld_dev bypasses the scalar DCache,
 * but the line itself may still be held stale, so it is invalidated first and
 * the invalidate is ordered before the read that follows.
 */
inline __aicore__ void topo_observe_cache_line(__gm__ volatile void *address) {
#if defined(__CCE_AICORE__)
    dcci(const_cast<__gm__ void *>(address), SINGLE_CACHE_LINE);
#else
    (void)address;
#endif
    topo_full_barrier();
}

/*
 * Make a whole span written by another processor visible, one line at a time.
 *
 * Needed for anything larger than a line that a peer published and this core
 * reads with ordinary cached loads -- the image's read-only tables, and the
 * image header itself, which is two lines wide. Invalidating only the first
 * line of a multi-line object is the silent-corruption case: the fields in the
 * uninvalidated remainder read stale, and on an image header those fields are
 * the offsets every other array is resolved through.
 *
 * `bytes` is rounded up to a whole line. This is a bring-up cost, paid once per
 * core, never on the task path.
 */
inline __aicore__ void topo_observe_range(__gm__ volatile void *address, uint64_t bytes) {
#if defined(__CCE_AICORE__)
    __gm__ uint8_t *base = reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ void *>(address));
    for (uint64_t offset = 0; offset < bytes; offset += CACHE_LINE_BYTES) {
        dcci(base + offset, SINGLE_CACHE_LINE);
    }
#else
    (void)address;
    (void)bytes;
#endif
    topo_full_barrier();
}

/*
 * Push this core's dirty lines for a span back to GM.
 *
 * A kernel's scalar GM stores land in this core's data cache. A consumer reads
 * the same addresses through ld_dev, which bypasses that cache, so without an
 * explicit writeback it reads pre-task bytes -- the counter says DONE and the
 * outputs are stale. The tile pipeline's own DMA writes do not need this; the
 * scalar ones do, and nothing distinguishes them at this level.
 */
inline __aicore__ void topo_publish_range(__gm__ volatile void *address, uint64_t bytes) {
#if defined(__CCE_AICORE__)
    __gm__ uint8_t *base = reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ void *>(address));
    for (uint64_t offset = 0; offset < bytes; offset += CACHE_LINE_BYTES) {
        dcci(base + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
#else
    (void)address;
    (void)bytes;
#endif
    topo_full_barrier();
}

/*
 * Write back everything this core has dirtied, then order it.
 *
 * A task publishes its completion counter without knowing which addresses its
 * kernel wrote, so a per-line writeback is not available to it and the whole
 * data cache is the only sound choice. This runs once per retired task, never
 * inside the wait, which is what makes the cost acceptable.
 */
inline __aicore__ void topo_publish_all() {
#if defined(__CCE_AICORE__)
    dcci(static_cast<__gm__ void *>(nullptr), ENTIRE_DATA_CACHE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/*
 * Observation load. Not an ownership operation -- the value it returns is a
 * fact about the past, and the caller is responsible for any barrier that makes
 * acting on it safe.
 */
inline __aicore__ uint64_t topo_gm_query(__gm__ volatile uint64_t &value) {
#if defined(__CCE_AICORE__)
    return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(&value), 0));
#else
    return __atomic_load_n(&value, __ATOMIC_ACQUIRE);
#endif
}

inline __aicore__ void topo_gm_store(__gm__ volatile uint64_t &value, uint64_t desired) {
#if defined(__CCE_AICORE__)
    st_dev(desired, const_cast<__gm__ uint64_t *>(&value), 0);
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_store_n(&value, desired, __ATOMIC_RELEASE);
#endif
}

inline __aicore__ uint64_t topo_gm_fetch_add(__gm__ volatile uint64_t &value, uint64_t delta) {
#if defined(__CCE_AICORE__)
    return atomicAdd(const_cast<__gm__ uint64_t *>(&value), delta);
#else
    return __atomic_fetch_add(&value, delta, __ATOMIC_ACQ_REL);
#endif
}

/*
 * Returns the value the location held before the call, so a caller detects it
 * won the race by comparing the result against `expected` -- which is how the
 * error latch keeps the first code rather than the last.
 */
inline __aicore__ uint64_t
topo_gm_compare_exchange(__gm__ volatile uint64_t &value, uint64_t expected, uint64_t desired) {
#if defined(__CCE_AICORE__)
    return atomicCAS(const_cast<__gm__ uint64_t *>(&value), expected, desired);
#else
    uint64_t observed = expected;
    __atomic_compare_exchange_n(&value, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    return observed;
#endif
}

/*
 * One pass of a spin.
 *
 * The barrier is what forces the next pass to re-read GM instead of spinning on
 * a value the compiler hoisted out of the loop. SPIN_WAIT_HINT() is the
 * platform's own per-variant hint and carries the rest: on silicon it expands
 * to ((void)0), so no task's latency pays for a yield that has no scheduler to
 * yield to, which is what codestyle rule 5 requires. Under __CPU_SIM an AICore
 * is an ordinary host thread competing for a host core, and the macro expands
 * to a yield so a spinning waiter cannot starve the very producer it waits on.
 */
inline __aicore__ void topo_spin_hint() {
    topo_full_barrier();
    SPIN_WAIT_HINT();
}

}  // namespace simpler::topo_queue
