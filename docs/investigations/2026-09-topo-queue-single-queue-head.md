# topo_queue's one global queue head caps dispatch at ~1 task per 3 µs

**Date**: 2026-09-30
**Verdict**: the scheme is dropped in this shape — a single CAS head is 5.9×
slower per task than `host_build_graph`'s resident scheduler on a5 silicon, and
the cost is contention for one cache line, not anything the kernels do. Worth
re-opening only with more heads (see *When to reconsider*).

## Question

`topo_queue` (branch `dev/clc-mpmc`) replaces `host_build_graph`'s device-side
scheduler with a decentralised pull: the AICPU sorts the graph once into a GM
task table and stops, and every AICore peeks a shared queue head, claims it by
CAS when the head task matches its own type, waits on its predecessors'
completion counters, runs, and publishes its own counter. No ready queue, no
per-task dispatch register write, ~470 lines in place of the resident
scheduler's ~10,000.

Under simulation it looked like a large win: same bgemm workload, ~0.88 s
against the resident scheduler's ~2.6 s. The question was whether that holds on
silicon.

## What was tried

`examples/a5/{host_build_graph,topo_queue}/benchmark_bgemm` are a matched pair:
byte-identical AIC/AIV/orchestration sources, identical params, and only
`@scene_test(runtime=)` differing. All runs on one a5 card, exclusive
`task-submit` lock.

1. **Clean timing.** `--rounds 100 --skip-golden`, no diagnostics, `device_wall`
   trimmed mean, parsed with `strace_timing --rounds-table`.
2. **Task-count sweep.** `Scale0..Scale3` cases (added by this work, manual on
   every platform) give 200 / 500 / 1000 / 2000 / 4000 tasks, 30 rounds each.
3. **Stage split.** `strace_timing --tree` for the `bind` / `publish_image` /
   `runner_run` / `device_wall` / `validate` breakdown.
4. **Per-task traces.** `--enable-chip-swimlane 2` plus a separate
   `--enable-dep-gen` run. `topo_queue` fills the resident scheduler's own
   `SchedulerTaskTrace` cells, so the host publication path already in
   `runtime_maker.cpp` emits `chip_swimlane_records.json` unchanged.
5. **HBG legacy** was measured by forcing the fallback with a temporary
   one-token edit in `create_scheduler_state` (`TaskKind::GRAPH || true`),
   reverted immediately after.

## Result

`device_wall`, µs, trimmed mean over 30 rounds:

| tasks | resident | legacy | topo_queue | topo/resident |
| ----: | -------: | -----: | ---------: | ------------: |
| 200 | 266.0 | 450.3 | 1034.2 | 3.89× |
| 500 | 415.8 | 768.6 | 2207.1 | 5.31× |
| 1000 | 694.9 | 1365.5 | 3920.2 | 5.64× |
| 2000 | 1224.8 | 2540.7 | 7265.6 | 5.93× |
| 4000 | 2305.5 | 4873.9 | 13206.5 | 5.73× |

All three are linear in task count, so the whole story is a per-task serial
cost: **0.538 µs (resident), 1.168 µs (legacy), 3.184 µs (topo_queue)**.

Per-task traces, 1000 tasks over 96 cores. The summed kernel time is the same
for all three (~20 ms), so the difference is entirely how well the cores are
kept fed:

| | span | cores | Σ kernel | core utilisation |
| --- | ---: | ---: | ---: | ---: |
| resident | 535.7 µs | 70 | 19 989 µs | 53.3% |
| legacy | 831.0 µs | 91 | 20 883 µs | 27.6% |
| topo_queue | 3110.9 µs | 96 | 20 223 µs | **6.8%** |

**Claims are strictly serial.** Claim order equals task id order for all 1000
tasks, and the inter-claim gap is 3.07 µs (p10 1.63 / p50 2.93 / p90 4.82),
constant across every sixth of the run. An average of 6.6 cores are executing;
the other ~89 are spinning on the head.

**The gap scales with how many cores are polling**, which is what identifies the
mechanism:

| next task's type | head dwell | free cores of that type |
| --- | ---: | ---: |
| AIC | 3.564 µs | 27 of 32 |
| AIV | 2.580 µs | 62.5 of 64 |

Two points fit `dwell ≈ 1.8 µs + 47 µs / N_polling`. A loop body of two GM reads
taking ~47 µs to come round says those two lines are saturated: every core
invalidates and re-reads the same head and the same run-control word on every
pass.

**Two attempts to reduce that traffic both made it worse**, from opposite
directions, which is the signature of sitting at a saturation point:

| change | device_wall | vs baseline |
| --- | ---: | ---: |
| baseline | 3990 µs | — |
| sample the error poll every 64th pass | 4721 µs | +18% |
| the above plus foreign-head backoff | 4914 µs | +23% |

Cheaper passes mean each core polls more often and the line gets hotter; backing
off means the core that should claim next notices later.

**End to end the picture is softer than the device numbers**, because this
benchmark re-binds every round and host graph construction dominates:

| stage | resident | legacy | topo_queue |
| --- | ---: | ---: | ---: |
| bind (host) | 10 540 | 9 931 | 9 876 |
| publish_image | 1 232 | 559 | 1 166 |
| device_wall | 695 | 1 363 | 3 922 |
| validate | 1 892 | 1 962 | 1 804 |
| **run total** | **16 521** | **15 693** | **18 663** |

5.6× on device is 13% end to end here. A workload that binds once and replays
would see the device figure instead.

Two side findings. `topo_queue` pays ~600 µs per run in `publish_image` for a
`SchedulerState` it never reads: it requires resident mode only to reach the
graph address in the bootstrap context. And the repo's `sched_overhead_analysis`
reports 0% overhead for it, which is not a compliment — that metric measures the
gap between *ready* and *dispatched*, and a pull scheduler claims before
readiness, so the gap is negative and invisible. The metric is push-shaped.

## Why not (now)

One CAS head is a hard throughput ceiling of about one task per 3 µs, and it
does not move with the amount of available work: bgemm's 500 GEMM tasks have no
predecessors at all and still come out one every ~3 µs, because the head cannot
be skipped — skipping is what the deadlock-freedom argument in `topo_worker.h`
forbids, since a consumed index nobody executes strands every consumer of that
task.

The resident scheduler has no such point: 32 cluster schedulers dispatch in
parallel, each to its own three lanes. That is why it wins, and why `topo_queue`
is slower even than AICPU-side legacy dispatch.

Independently, a task occupies a core for ~16 µs waiting on predecessors before
it runs, because this scheme claims first and waits second. With 96 cores that
is hidden by parallelism; with a tight core budget it would not be.

## When to reconsider

- **More heads.** The measured dwell depends on the number of cores polling one
  line, so per-type queues (two heads) or per-cluster queues are the direct
  lever, and the two-queue variant already exists in this branch's history —
  `worker_step` is parameterised over `(order, count, head)`. Two heads also
  advance in parallel, which is the larger effect. Reaching resident's numbers
  would need heads at cluster scale, which is approximately resident's own
  structure.
- **A workload that binds once and replays many times**, where `device_wall` is
  the end-to-end number rather than 4% of it.
- Not on the strength of a simulation result. Sim's `dcci` is a full memory
  fence, it has no GM latency, and 24 threads oversubscribed on ~10 host cores
  cannot produce 96-core head contention — it reported this scheme 3× *faster*
  than the resident scheduler.

## Reproducing

```bash
# clean timing (no diagnostics; --rounds > 1 disables them anyway)
python examples/a5/<runtime>/benchmark_bgemm/test_benchmark_bgemm.py \
    -p a5 -d $TASK_DEVICE --rounds 100 --skip-golden --case Case0 | tee run.log
python -m simpler_setup.tools.strace_timing run.log --rounds-table
python -m simpler_setup.tools.strace_timing run.log --tree

# per-task traces (single round, two separate runs)
python .../test_benchmark_bgemm.py -p a5 -d $TASK_DEVICE --rounds 1 \
    --skip-golden --case Case0 --enable-chip-swimlane 2
python .../test_benchmark_bgemm.py -p a5 -d $TASK_DEVICE --rounds 1 \
    --skip-golden --case Case0 --enable-dep-gen
python -m simpler_setup.tools.sched_overhead_analysis \
    --chip-swimlane-records-json outputs/<sw-run>/chip_swimlane_records.json \
    --deps-json outputs/<dep-run>/deps.json
```

`Scale0..Scale3` carry the sweep's other task counts and are manual on every
platform; select them with `--case ScaleN --manual include`.
