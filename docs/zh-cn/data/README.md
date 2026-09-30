# 实测原始数据

[topo_queue_experiments.md](../topo_queue_experiments.md) 用到的原始采集数据。留档的原因很实际：**这是 bgemm 泳道的唯一副本**——服务器上的 `outputs/` 目录在 2026-09-30 晚间断电后未能取回，文档里的图和表都是从这几个文件算出来的。

## 文件

| 文件 | 来源 | 内容 |
| --- | --- | --- |
| `bgemm-swimlane-resident.json` | HBG resident | 原始 `chip_swimlane_records.json` |
| `bgemm-swimlane-legacy.json` | HBG legacy（临时强制回退） | 同上 |
| `bgemm-swimlane-topo.json` | topo_queue | 同上 |
| `bgemm-lanes.json` | 上面三份加工而来 | 按核归并的泳道数据，画图直接用 |

三次采集都是 `benchmark_bgemm` 的 `Case0`（1000 个任务），`--rounds 1 --skip-golden --enable-chip-swimlane 2`，同一张 A5 卡，96 个核（32 AIC + 64 AIV）。

## 数据格式

`chip_swimlane_records.json` 的两个数组是**位置参数**，不是键值对：

```text
aicore_tasks      [core, task_id, task_id, kernel_start, kernel_end, receive_to_start, run_epoch]
scheduler_tasks   {"producer": ..., "records": [[core, task_id, dispatch_end, complete_start, run_epoch]]}
```

时间戳单位是 AICore 系统计数器的 tick，`metadata.clock_freq_hz` 给出频率（实测 1 GHz，所以 1 tick = 1 ns）。`metadata.core_types` 按**稠密 worker 索引**排列，`aicore_tasks` 里的 `core` 就是这个索引。

三个时间点的语义，三种 runtime 是对齐的：

| 字段 | resident / legacy | topo_queue |
| --- | --- | --- |
| `dispatch_end` | 调度器写下派发描述符 | 核 CAS 抢到队头 |
| `kernel_start` | 核开始跑 kernel | 前驱全部就绪、核开始跑 |
| `kernel_end` | kernel 返回 | kernel 返回 |

所以 `kernel_start − dispatch_end` 在 topo_queue 上就是"已认领但还在等前驱"的时长，在 resident 上是早发提前量，在 legacy 上恒为 0。

`bgemm-lanes.json` 是加工过的形式，每个核一个数组，每项 `[dispatch, start, end]`，单位微秒，已减去该次运行的起点。

## 怎么用

```python
import json
d = json.load(open("bgemm-swimlane-topo.json"))
at = d["aicore_tasks"]
disp = {r[1]: r[2] for r in d["scheduler_tasks"]["records"]}
t0 = min(r[3] for r in at)

# 认领间隔：队头每推进一格要多久
claims = sorted((disp[r[1]] - t0) / 1000 for r in at)
gaps = [claims[i+1] - claims[i] for i in range(len(claims) - 1)]
```

仓库自带的工具也能直接吃这些文件：

```bash
python -m simpler_setup.tools.swimlane_converter <file>            # 转 Perfetto JSON
python -m simpler_setup.tools.sched_overhead_analysis \
    --chip-swimlane-records-json <file> --deps-json <deps.json>    # 需要另一次 dep_gen 运行
```

`sched_overhead_analysis` 需要配套的 `deps.json`，那个没有留档——它来自另一次单独运行，同样丢在断电的服务器上。文档第 5.4 节引用的调度开销百分比是当时算好抄下来的。

## 没有留档的

- 四种 DAG 形状（`multi_core_dag`）的 24 次采集，全部在服务器上；
- 所有 `deps.json`；
- 干净性能数字那批（`--rounds 30/100`）的原始日志——那些跑时不开诊断，本来也只产出 `[STRACE]` 文本，结论已抄进文档。
