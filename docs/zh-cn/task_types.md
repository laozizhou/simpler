# simpler 任务类型与用例统计

**日期**：2026-10-08 · **代码版本**：`dev/clc-mpmc` 分支 `281428cb` · **范围**：`examples/` 与 `tests/st/` 下全部 TRB、HBG 用例

结论：

- 仓库共 211 个用例，最常见的是单 block 的 AIV 和 AIC 任务。只用这两种任务的用例占 48%。
- 两个模型负载 deepseek 和 qwen3 以 SPMD 为主，并且都用到了 DUMMY、ALLOC 和三核 MIX。qwen3 还用了 MIX 加 sync_start，deepseek 还用了谓词派发，HBG 版本还用了 GRAPH。
- 双核 MIX 和 AIV 加 sync_start 只在功能测试里出现。AIC 加 sync_start 和 AIC+AIV1 组合没有任何用例。

---

## 1. 任务类型

### 1.1 种类

| 种类 | 提交接口 | 含义 | TRB | HBG |
| --- | --- | --- | --- | --- |
| 计算任务 | `rt_submit_aic_task`、`rt_submit_aiv_task`、`rt_submit_task` | 上核执行 kernel，细分见 1.2 | 有 | 有 |
| DUMMY | `rt_submit_dummy_task` | 只参与依赖，就绪后由调度器直接完成 | 有 | 有 |
| ALLOC | `alloc_tensors` | 只分配输出 buffer，提交时由编排器直接完成 | 有 | 有 |
| GRAPH | `rt_submit_graph` | 容器：占一个任务槽，内部是一张录好的子任务图，录一次后可多次回放 | 无 | 有 |

DUMMY 和 ALLOC 没有计算量，但模型负载里也在用：ALLOC 预先分配中间张量，DUMMY 做阶段栅栏。GRAPH 的细节见 [GRAPH_EXECUTION.md](../../src/common/host_build_graph/docs/GRAPH_EXECUTION.md)。

### 1.2 计算任务的两个维度

**形态：每个 block 占什么核。** 调度器按 AIC、AIV、MIX 三条就绪队列排队，哪类核空了就去哪条队列取任务。MIX 的四种核组合共用一条队列，派发时都是找一个所需核都空闲的 cluster。定义见 [submit_types.h](../../src/a5/runtime/tensormap_and_ringbuffer/runtime/submit_types.h) 的 `ResourceShape`——枚举里还有第四个值 DUMMY，它有自己的就绪队列，就绪后由调度器直接完成、不派发到核，所以不算在可派发的三条里。

| 就绪队列 | 核组合 |
| --- | --- |
| AIC | 一个 AIC |
| AIV | 一个 AIV |
| MIX | 同一 cluster 内的 AIC+AIV0、AIC+AIV1、AIV0+AIV1 或 AIC+AIV0+AIV1，各核可以跑不同 kernel |

**启动方式：跑几份、是否同时起跑。**

| 启动方式 | 设置 | 行为 |
| --- | --- | --- |
| 单 block | 默认 | 占一份核 |
| SPMD | block 数大于 1 | 同一个 kernel 复制 N 份，有空闲核就派，可以分批跑 |
| SPMD 加 sync_start | 再打开 `require_sync_start` | N 份占齐核后一起起跑，N 不能超过该类核的总数 |

sync_start 只在 block 数大于 1 时生效。SPMD 和 MIX 都会占多个核，区别在于：SPMD 的各份是同一个 kernel，可以放在任意空闲核上先后执行；MIX 的各核必须在同一个 cluster 内一起派发。

两个维度组合起来，计算任务有 3 × 3 = 9 种。加上 DUMMY、ALLOC，TRB 共 11 种；HBG 再加 GRAPH，共 12 种。

### 1.3 属性

以下三项不改变任务占用的核，只算属性：

- **谓词派发**：派发时检查一个张量元素，不满足就跳过执行，按 DUMMY 的方式完成。
- **allow_early_resolve**：打在**生产者**身上，表示它的消费者可以在它完成前就被放到核上。没打这个标志的生产者会取消其消费者的提前派发资格。
- **timing slot**：给任务打计时标签，用于性能分析。

---

## 2. 统计口径

- **用例**：`examples/` 与 `tests/st/` 下声明了 TRB 或 HBG 的每个 `test_*.py`，共 211 个。不含 `tests/ut`。
- **arch 和 runtime**：按用例声明归属。同时声明两种 arch 或两种 runtime 的用例在多列都计数，所以各列之和大于"合计"。
- **类型判定**：对用例实际编译的编排源码逐个提交调用做静态扫描。block 数用变量传入时按 SPMD 计。
- **计数**：一个用例用到几种类型，每种各计一次。
- **局限**：只看源码里是否出现，不看出现次数。故意触发报错的负面用例也计入。

---

## 3. 统计结果

### 3.1 用例数量

| 分组 | a2a3 TRB | a2a3 HBG | a5 TRB | a5 HBG | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 用例总数 | 98 | 38 | 84 | 28 | 211 |
| 其中 examples | 33 | 4 | 26 | 3 | 54 |
| 其中 tests/st | 65 | 34 | 58 | 25 | 157 |
| 其中不提交芯片任务 | 5 | 0 | 5 | 1 | 7 |
| 提交芯片任务的用例 | 93 | 38 | 79 | 27 | 204 |

不提交芯片任务的 7 个用例是 Worker 生命周期、内存操作、prewarm 配置等演示或测试，以下各表不计入。

### 3.2 各类型出现在多少个用例中

形态一栏按核组合细分，所以 MIX 占四行；1.2 节的「3 种形态」和下面 3.3 节都把这四种收回成一个 MIX。

| 维度 | 类型 | a2a3 TRB | a2a3 HBG | a5 TRB | a5 HBG | 合计 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 种类 | 计算任务 | 92 | 37 | 78 | 26 | 203 |
| 种类 | DUMMY | 8 | 5 | 8 | 4 | 21 |
| 种类 | ALLOC | 12 | 7 | 12 | 6 | 35 |
| 种类 | GRAPH | 0 | 8 | 0 | 6 | 14 |
| 形态 | AIC | 23 | 18 | 23 | 14 | 75 |
| 形态 | AIV | 75 | 29 | 60 | 21 | 156 |
| 形态 | MIX：AIC+AIV0+AIV1 | 16 | 11 | 15 | 7 | 44 |
| 形态 | MIX：AIC+AIV0 | 2 | 0 | 2 | 0 | 4 |
| 形态 | MIX：AIV0+AIV1 | 1 | 0 | 1 | 0 | 2 |
| 形态 | MIX：AIC+AIV1 | 0 | 0 | 0 | 0 | 0 |
| 启动方式 | 单 block | 77 | 30 | 65 | 22 | 165 |
| 启动方式 | SPMD | 11 | 9 | 9 | 6 | 33 |
| 启动方式 | SPMD 加 sync_start | 10 | 6 | 10 | 4 | 29 |
| 属性 | 谓词派发 | 2 | 4 | 1 | 2 | 9 |
| 属性 | allow_early_resolve | 5 | 7 | 4 | 3 | 19 |
| 属性 | timing slot | 1 | 1 | 1 | 1 | 1 |

### 3.3 形态 × 启动方式

| 形态 × 启动方式 | a2a3 TRB | a2a3 HBG | a5 TRB | a5 HBG | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AIC 单 block | 21 | 14 | 21 | 13 | 66 |
| AIC SPMD | 4 | 6 | 3 | 3 | 16 |
| AIC SPMD 加 sync_start | 0 | 0 | 0 | 0 | 0 |
| AIV 单 block | 68 | 27 | 53 | 20 | 140 |
| AIV SPMD | 6 | 5 | 6 | 3 | 18 |
| AIV SPMD 加 sync_start | 4 | 0 | 4 | 0 | 8 |
| MIX 单 block | 5 | 2 | 7 | 1 | 12 |
| MIX SPMD | 5 | 3 | 2 | 2 | 11 |
| MIX SPMD 加 sync_start | 7 | 6 | 7 | 4 | 23 |

要点：

- **MIX 几乎全是三核。** 双核组合只出现在 TRB 的 `mixed_example` 和 `chip_swimlane` 两个功能测试里。
- **sync_start 主要和 MIX 搭配。** AIV 加 sync_start 只在 TRB 里出现，AIC 加 sync_start 没有出现。
- **HBG 没有覆盖**双核 MIX 和 AIV 加 sync_start。

---

## 4. 模型负载

功能测试会刻意覆盖少见的类型，更能代表实际需求的是两个模型 decode 负载，共 6 个变体（`examples/` 下还有两个 a2a3 的 `qwen3_14b_serving_effective`，是 serving 负载不是 decode，未计入）。下表只统计种类和形态 × 启动方式的提交调用点数，属性不在表内、见表后正文。很多调用点在循环里，所以运行时的任务数更多。全为 0 的行已省略。

| 提交调用点 | deepseek a2a3 HBG | deepseek a2a3 TRB | qwen3 a2a3 HBG | qwen3 a2a3 TRB | qwen3 a5 HBG | qwen3 a5 TRB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIC 单 block | 1 | 2 | 4 | 4 | 4 | 4 |
| AIC SPMD | 54 | 78 | 14 | 14 | 14 | 14 |
| AIV 单 block | 67 | 95 | 15 | 15 | 15 | 15 |
| AIV SPMD | 120 | 169 | 2 | 2 | 2 | 2 |
| MIX SPMD | 8 | 12 | 0 | 0 | 0 | 0 |
| MIX SPMD 加 sync_start | 0 | 0 | 1 | 1 | 1 | 1 |
| DUMMY | 7 | 10 | 13 | 13 | 13 | 13 |
| ALLOC | 101 | 144 | 2 | 3 | 2 | 3 |
| GRAPH | 8 | 0 | 1 | 0 | 1 | 0 |

- **SPMD 是主力。** deepseek 的 AIC 调用点几乎全是 SPMD，block 数集中在 8、16、32。
- **MIX 全是三核**，而且都带多个 block。qwen3 唯一的 sync_start 用在 attention 任务上。
- **deepseek 两个变体都用了谓词派发**，挂在 MoE 专家相关的 SPMD 任务上。
- **没有模型负载用到**双核 MIX、单 block 的 MIX，或 AIC、AIV 加 sync_start。

---

## 5. 类型依赖

把类型按下面的顺序逐级累加，统计所用类型全部落在集合内的用例数。allow_early_resolve 和 timing slot 不影响能否运行，不计入。

| 累加到 | a2a3 TRB | a2a3 HBG | a5 TRB | a5 HBG | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 提交芯片任务的用例 | 93 | 38 | 79 | 27 | 204 |
| ① AIC、AIV 单 block | 56（60%） | 15（39%） | 42（53%） | 9（33%） | 98（48%） |
| ② 加 DUMMY、ALLOC | 70（75%） | 21（55%） | 56（71%） | 15（56%） | 134（66%） |
| ③ 加 MIX 单 block | 75（81%） | 23（61%） | 63（80%） | 16（59%） | 146（72%） |
| ④ 加 SPMD | 81（87%） | 26（68%） | 68（86%） | 19（70%） | 162（79%） |
| ⑤ 加 sync_start | 91（98%） | 28（74%） | 78（99%） | 20（74%） | 184（90%） |
| ⑥ 加谓词派发 | 93（100%） | 30（79%） | 79（100%） | 21（78%） | 190（93%） |
| ⑦ 加 GRAPH | 93（100%） | 38（100%） | 79（100%） | 27（100%） | 204（100%） |

模型负载最早在第⑤级才能满足：qwen3 TRB 在第⑤级，deepseek TRB 在第⑥级，三个 HBG 变体在第⑦级。
