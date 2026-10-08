# simpler 仓库总览

面向要在本仓库上开发新特性的人。讲清楚：这是什么、为什么这么设计、代码怎么组织、
在 macOS 上怎么搭环境、跑起来之后怎么用。

英文文档索引见 [docs/README.md](../README.md)。本文不重复那里的内容，只做导航与补充。

---

## 1. 背景：它在技术栈的哪一层

`simpler` 是昇腾 NPU 的**任务图运行时**。它在 `hw-native-sys` 栈里的位置：

```text
pto-isa  ──▶  simpler  ──▶  pypto  ──▶  pypto-lib  ──▶  pypto-serving
(tile ISA)    (运行时)      (编译器)     (算子/模型)      (推理服务)
                 ▲
              PTOAS (汇编器，全局安装)
```

**它负责**：任务编排与分发 —— DAG 提交、AICPU/主机调度器、Worker 与设备生命周期、
host↔device 握手协议，以及读回这些信息的 DFX/profiling 工具链。

**它不负责**：kernel 函数体、tile ISA、图编译、模型与服务逻辑。

---

## 2. 问题：为什么需要一个专门的运行时

目标硬件是昇腾 NPU（a2/a3 与新一代 a5）：

```text
Host (CPU + DDR) ──PCIe/UB──▶ Ascend chip
                               ├─ AICPU 集群（控制）
                               ├─ AICore 集群
                               │   ├─ AIC = CUBE，矩阵运算
                               │   └─ AIV = VECTOR，向量运算
                               └─ GM（共享 HBM）
```

三个难点决定了整体设计：

**一、主机往返太贵。** host↔chip 是微秒级延迟。实测每次 `run()` 约 43 µs 固定开销，
外加 8 µs/任务的边际开销（见 `docs/investigations/2026-07-host-dispatch-latency-budget.md`）。
固定开销会被大图摊薄，边际开销不会 —— 上千个任务就是毫秒量级的纯调度成本。
**于是把调度搬到片上** —— 这是本仓库最核心的取舍。

**二、片上通信原语极其有限，且踩错会挂死整颗芯片。** 详见
[`.claude/rules/ascend.md`](../../.claude/rules/ascend.md)：AICore 无法写 `DATA_MAIN_BASE`；
AICore 的 LSU 够不到 SPR MMIO 窗口；AICPU 单线程 LDR 严格串行（约 95 ns/次，无法流水）。
这些约束是调度器设计复杂度的根源。

**三、故障信号是模糊的。** 主机侧 `507018` 是泛化错误码，底下可能是结构性死锁、容量耗尽、
自旋超时，或仅仅是"算得慢"。因此仓库在可观测性上投入很重（见 §7）。

---

## 3. 做了什么：四个维度

### 3.1 维度总览

| 维度 | 取值 | 位置 |
| --- | --- | --- |
| **架构** | `a2a3` / `a5` | `src/{arch}/` |
| **平台变体** | `onboard`（真硬件）/ `sim`（线程模拟） | `src/{arch}/platform/{variant}/` |
| **运行时** | `host_build_graph` / `tensormap_and_ringbuffer` | `src/{arch}/runtime/{runtime}/` |
| **层级** | `core=0, chip=2, node=3, network1..3=4..6` | `python/simpler/worker_level.py` |

前三个维度是笛卡尔积：**2 × 2 × 2 = 8 种构建组合**，每组合编 4 个目标
（`aicore` / `aicpu` / `host` / `orchestration`）。产物落在
`build/lib/{arch}/{variant}/{runtime}/`。

### 3.2 两个运行时：区别是"图在哪儿建"

| | `tensormap_and_ringbuffer` (TRB) | `host_build_graph` (HBG) |
| --- | --- | --- |
| 建图位置 | **AICPU（设备侧）** | 主机 CPU |
| orchestration `.so` | 设备上 dlopen，**与调度并发** | 主机 dlopen，跑到结束再一次性上传 |
| 线程布局 | a2a3：1 orch + 3 sched；a5：1 orch + 4 sched | 设备侧无 orchestrator，全部线程都调度 |
| 内存模型 | 4 组环形缓冲，**边跑边回收**（ring 索引 = `min(scope_depth, 3)`） | bump 分配器只进不退，全图必须装得下 |
| 背压 / 流控 | 有 | 无 |
| 典型用途 | 生产负载 | 开发、调试 |

**两者都用 TensorMap 做依赖推断** —— 用户不声明依赖，运行时从张量地址重叠自动推断。
区别在于 TRB 的 `ChipTensorMap` 活在设备 GM 里，HBG 的 TensorMap 是纯主机对象、从不上设备。

HBG 独有 **Graph Execution**：把子 DAG 录制一次、缓存最多 16 个 Definition，之后只发一个
`GRAPH` 任务让设备调度器自己展开。

> 切换方式只有一个字符串：`@scene_test(runtime="...")` 或 `Worker(runtime=...)`。
> 但 orchestration 源码在两个运行时下 API 不同，所以示例树按 runtime 分目录。

### 3.3 L0–L6 层级模型

```text
L6 network3 / Cluster      L3 node    / Node      ← 单主机（多芯片 + SubWorker）
L5 network2 / SuperNode    L2 chip    / Processor ← 一颗 NPU  ★ 分界线
L4 network1 / Pod          L1 —       / L2Cache
                           L0 core    / AIC, AIV
```

**L2 是分水岭**：L0–L2 在设备上（靠 GM 原子操作和 barrier），L3–L6 在主机/集群
（同一套引擎递归组合，靠 fork + 共享内存 mailbox）。

L3 以上每层跑同样三个组件：

- **Orchestrator** —— 单线程建图，拥有 `Ring`（带背压的 slot 池）、`TensorMap`、`Scope`
- **Scheduler** —— 独立线程，管 directed NEXT_LEVEL / shared SUB / completion 三类队列；**自己从不执行用户代码**
- **WorkerManager** —— 每个 worker 一条 endpoint lane，经共享内存 mailbox 通信

> 四类调度器（L3+ 主机 / L2 TRB AICPU / L2 HBG AICPU / a5 HBG 常驻 AIV，算上兄弟架构的克隆
> 共 6 份实现）的实现细节、判断点与扩展点，见 [zh-cn/schedulers.md](schedulers.md)。

命名规范（见 `.claude/rules/codestyle.md` 规则 13）：按**角色**不按层号。
L3+ 无前缀（`Worker`/`Tensor`），L2 用 `Chip` 前缀（`ChipWorker`/`ChipTensor`），
L0 用 `Core` 前缀（`CoreCallable`）。禁止 `L2Tensor` 这类名字。

### 3.4 三程序模型与 AICore 握手

一次计算涉及三个独立编译的程序：

| 程序 | 跑在哪 | 产物 |
| --- | --- | --- |
| Host runtime | 主机 CPU | `.so` |
| AICPU kernel（调度器） | 片上 AICPU | `.so` |
| AICore kernel（计算） | 片上 AIC/AIV | `.o` |

派发走寄存器握手：

```text
AICore: 报 aicore_done → 等 DATA_MAIN_BASE != 0
AICPU:  往该核的 DATA_MAIN_BASE 写单调递增序号
AICore: 读双缓冲 DispatchPayload（payload + (id & 1)）
        → 调 function_bin_addr(args) → 往 COND 写 ACK 和 FIN
AICPU:  轮询 COND 收 FIN → 释放 fanout → 推进水位
```

双缓冲让 AICPU 能在 AICore 还在算上一个任务时预写下一个。

### 3.5 a2a3 vs a5 的实质差异

| | a2a3 | a5 |
| --- | --- | --- |
| 核数 | 1 die：24 AIC + 48 AIV = 72 | 2 die 合成 1 device：36 AIC + 72 AIV = 108 |
| AICPU 线程 | 4（1 orch + 3 sched） | 5（1 orch + 4 sched） |
| 系统计数器 | 50 MHz | 1 GHz |
| 缓存 | 主机 DMA/SDMA 写后需显式 invalidate | DMA 与 AICPU 一致，无需 |
| 异步后端 | SDMA | SDMA（默认）+ RDMA / URMA（各自由一个 cmake 选项门控，均默认关） |
| **HBG 调度** | AICPU 调度 | **常驻 AIV 调度器** —— AICore 自己解依赖、经 SSBUF 传递派发状态，AICPU 只做初始化/监控/拆除；原 AICPU 调度路径保留为显式回退（`aicore_legacy_executor.cpp`） |

最后一行是 a5 最大的架构跃迁。注意：**那个 AIV 调度器是普通 C++ 控制逻辑，不含任何 tile 指令**，
所以它和计算 kernel 的编译依赖完全不同（见 §5.3）。

---

## 4. 代码放在哪里

### 4.1 顶层

| 目录 | 文件数 | 内容 |
| --- | --- | --- |
| [`src/`](../../src/) | 544 | C++ 核心。`common/` 269、`a2a3/` 128、`a5/` 147 |
| [`examples/`](../../examples/) | 856 | 53 个可运行示例目录 |
| [`tests/`](../../tests/) | 877 | `st/` 场景测试、`ut/py` + `ut/cpp` 单测、`lint/` |
| [`docs/`](../README.md) | 126 | 按"你想干什么"分组 |
| [`simpler_setup/`](../../simpler_setup/) | 41 | **用户可见**的构建/测试脚手架 |
| [`python/`](../../python/) | 26 | `simpler/` 包 + `bindings/` nanobind 扩展 |
| [`tools/`](../../tools/) | 51 | 独立 CANN/ACL 参考工程 + benchmark 脚本 |
| [`.claude/`](../../.claude/) | 54 | 11 条常驻规则 + 28 个技能 |

规模约 90 万行：C++ 66 万、Python 16 万、Markdown 5.9 万。

### 4.2 `src/` 内部

```text
src/
├── common/                        跨架构共用（269 文件）
│   ├── platform/     138          平台抽象：onboard 包 CANN，sim 是线程模拟，include 是共用接口 + DFX 采集器
│   ├── host_build_graph/  42      HBG 共用代码（.cpp 按"哪个目标编译它"分进 host/ device/ shared/）
│   ├── hierarchical/  21          L3+ 引擎：Orchestrator / Scheduler / WorkerManager / Ring / TensorMap
│   ├── task_interface/ 17         host↔device 数据词汇表：ChipCallable / CallConfig / TaskArgs / ChipTensor
│   ├── worker/        12          ChipWorker、ChipRunLane、PipelineSlotPool、runtime_c_api.h（C ABI）
│   ├── tensormap_and_ringbuffer/ 6  TRB 共用代码
│   ├── log/ utils/ aicpu_loader/ platform_comm/ runtime_status/
├── a2a3/  128
└── a5/    147
    ├── docs/                      per-arch 文档
    ├── platform/{include,onboard,sim,shared}/
    └── runtime/{host_build_graph,tensormap_and_ringbuffer}/
        ├── build_config.py        四个编译目标的 include/source 目录
        ├── aicore/ aicpu/ host/ orchestration/ runtime/ common/ docs/
```

**共享结构是"十字形"的**：沿架构轴共享 `common/{host_build_graph,tensormap_and_ringbuffer}/`，
沿运行时轴共享 `common/{platform,hierarchical,worker,...}/`。
真正私有的 runtime 代码约 7.9 万行，共享约 8.9 万行。

一个值得注意的不对称：**HBG 已抽出 42 个文件到 `common/`，TRB 只抽了 6 个** ——
所以 TRB 的 a2a3/a5 两棵树是近似拷贝（51 个同名文件里 23 个字节完全相同）。
改 TRB 时务必同步两边（`.claude/rules/codestyle.md` 规则 10 明确要求）。

### 4.3 几个特别的目录

- **[`docs/investigations/`](../investigations/README.md)** —— 28 篇"考虑过但否决"。
  **提优化方案前先 grep 这里**，避免重新推导已被否决的结论。
- **[`.claude/rules/`](../../.claude/rules/)** —— 11 条常驻规则，读起来像踩坑史。
  尤其 `comments.md`（注释只写"现在是什么"）、`codestyle.md`（13 条）、
  `running-onboard.md`（共享机器必须走 `task-submit` 抢设备锁）。
- **[`tools/cann-examples/`](../../tools/cann-examples/)** —— 硬件微基准，是那些硬件结论的实验台。
- **[`docs/dfx/`](../dfx/README.md)** + `simpler_setup/tools/` —— 自带的性能分析闭环（见 §7）。

---

## 5. 在 macOS 上搭环境

macOS 只能跑 **sim**（`a2a3sim` / `a5sim`），不能跑真硬件 —— 那需要 CANN 工具链和昇腾卡。
但 sim 覆盖了绝大部分功能验证。

> **已实测**（macOS 26 / Apple Silicon / 8 核）：`tests/ut/cpp` 251 个 C++ 单测全通过；
> `--platform a2a3sim` 全量 92 个用例零失败。
>
> **a5sim 在 8 核机器上要额外两条措施**，否则会出现大批"失败"或整体卡死 ——
> 两者都是宿主资源问题，不是代码缺陷：
>
> 1. **必须 `export OMP_NUM_THREADS=1`**。a5 每个虚拟设备要模拟 108 个 AICore 线程
>    + 14 个 AICPU 线程，叠加 OpenMP 的 per-thread TLS key 后耗尽进程线程资源，
>    报 `OMP: System error #35: Resource temporarily unavailable` / `pthread_key_create failed`。
> 2. **分目录跑，不要一次性全量**。即使设了上一条，单个 pytest 进程连续跑几十个
>    a5 用例后仍会因资源累积而崩溃（`Fatal Python error: Aborted`）。
>    按测试目录逐个起进程即可：实测 **71 个目录中 70 个通过、73 个用例零失败**，
>    而同样的用例在单进程全量模式下会在第 23 个之后连续失败。
>
> 唯一的例外是 `tests/st/a5/tensormap_and_ringbuffer/` 根目录下的三个 L3 文件，
> 它们会以约 615% CPU 空转 45 分钟无进展 —— 即
> [sim 过订阅活锁](../troubleshooting/sim-oversubscription-hang.md)。
> 这批 L3 覆盖在调度器的 Resource 阶段已验证（81 个 job 零失败），所以是宿主容量问题。
>
> a2a3sim（72 核）不受这两条影响，可以直接全量跑。

### 5.1 前置

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| Python | **≥ 3.10** | 系统自带 3.9 不可用，见 §5.4 |
| cmake | ≥ 3.20 | 可经 pip 装进 venv |
| Apple Clang | 系统自带 | 编运行时、orchestration |
| **GNU g++-15** | **必须真 GCC** | 编 sim kernel，Apple Clang 不行，见 §5.3 |
| 网络 | 需要 | 首次运行自动 clone pto-isa（81 MB） |

### 5.2 完整步骤

```bash
# 1. Python 3.10（系统只有 3.9 时）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.10

# 2. GNU GCC 15（编译 sim kernel 必需）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install gcc@15

# 3. venv
cd <repo>
uv venv --python 3.10 --system-site-packages .venv
.venv/bin/python -m ensurepip --upgrade        # uv 建的 venv 不带 pip

# 4. 构建依赖
.venv/bin/python -m pip install "scikit-build-core>=0.10.0" "nanobind>=2.0.0,<3" "cmake>=3.20"

# 5. 编译安装 —— 注意 PATH，构建子进程要找得到 cmake
export PATH="$PWD/.venv/bin:/opt/homebrew/bin:$PATH"
.venv/bin/python -m pip install --no-build-isolation -e .

# 6. 测试依赖
.venv/bin/python -m pip install pytest pytest-timeout pytest-xdist PyYAML torch numpy

# 7. 运行期环境变量（跑 a5sim 必设，见 §5.1 注记）
export OMP_NUM_THREADS=1

# 8. 验证（逐级加码）
.venv/bin/python examples/workers/l2/hello_worker/main.py -p a2a3sim -d 0   # 只测生命周期，不需 kernel
.venv/bin/python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
    -p a2a3sim --manual include                                             # 真实计算 + golden 校验
```

首次运行 `vector_example` 会自动 clone pto-isa 并编译 kernel，需要几分钟；之后走缓存。

**三个容易踩的坑：**

1. **`cmake` 必须在 `PATH` 上** —— pip 装的 cmake 在 `.venv/bin/`，但构建子进程不自动继承，
   会报 `CMake configuration not found`。
2. **uv 建的 venv 没有 pip** —— 需 `python -m ensurepip --upgrade`，之后用 `python -m pip`。
3. **pto-isa 首次 clone 很慢**（81 MB）。中途打断会留下残缺目录；可以先手工 clone：
   `git clone --no-checkout https://github.com/hw-native-sys/pto-isa.git build/pto-isa`
   然后 `git -C build/pto-isa checkout $(cat pto_isa.pin)`。

### 5.3 为什么必须是真 GNU GCC

Apple Clang 编不了 pto-isa 的 CPU 实现，实测两处硬错误：

```text
pto/common/utils.hpp:48   invalid input constraint 'l' in asm   ← GCC 特有的内联汇编约束
pto/cpu/ffts.hpp:48       no member named 'atomic_ref'          ← Apple libc++ 未实现 C++20 std::atomic_ref
```

第二条不是加编译选项能绕过的。

**但注意范围**：只有**计算 kernel** 需要 g++-15。三层编译分工如下：

| 编什么 | sim 用 | 真硬件用 |
| --- | --- | --- |
| host runtime / AICPU 调度器 / **a5 的 AIV 调度器** | Apple Clang | aarch64 交叉编译器 |
| orchestration `.so` | Apple Clang（`HOST_GXX`） | aarch64 交叉编译器 |
| **计算 kernel** | **g++-15**（`HOST_GXX_15`） | `ccec` |

分界线不是"跑在哪个核"，而是**代码里有没有 tile 指令**（`TLOAD`/`TMATMUL`/`TADD`）。
tile 指令需要 pto-isa 头文件；pto-isa 有 `cpu/`（15k 行，sim 用）和 `npu/`（78k 行，真硬件用）
两套实现，分流点在 `pto-inst.hpp` 的 `#if defined(__CPU_SIM)`。

a5 的 AIV 调度器虽然跑在 AICore 上，但 tile 指令出现 0 次，所以用 Apple Clang 就能编。

### 5.4 Python 版本的坑

仓库 `pyproject.toml` 声明 `requires-python = ">=3.9"`，并且为保 3.9 刻意把 nanobind 压在 `<3`。
但 `simpler_setup/scene_test.py` 用了 `zip(..., strict=True)`（需 3.10+），而 CI 的
`python-version` 全是 `["3.10"]`，所以声明的下限从未被测试。

**实践结论：用 3.10+。** 3.9 下场景测试会在编译 callable 时抛
`TypeError: zip() takes no keyword arguments`。

---

## 6. 跑起来之后怎么用

### 6.1 两种入口

**A. `@scene_test`（推荐）** —— 框架管编译、构参、golden 比对、多平台用例矩阵。

**B. 裸 `Worker` API** —— 什么都不隐藏，适合理解全流程或做非常规控制。
最值得读的单个文件是 [`examples/workers/l2/vector_add/main.py`](../../examples/workers/l2/vector_add/main.py)，
它把编译 → 包装 callable → 分配 → 拷贝 → 运行 → 校验逐步写开。

```python
worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
handle = worker.register(chip_callable)   # 必须在 init() 之前
worker.init()
try:
    buf = worker.malloc(nbytes)           # L2 专用；L3+ 用 alloc_child_tensor
    worker.copy_to(buf, host_tensor)
    args = TaskArgs()
    args.add_tensor(buf.tensor(shapes=..., dtype=DataType.FLOAT32), TensorArgType.INPUT)
    worker.run(handle, args, CallConfig())   # 阻塞；非阻塞用 submit() -> RunHandle
    worker.copy_from(host_out, out_buf)
finally:
    worker.close()                        # 放 finally，否则设备不释放
```

要点：**注册在 `init()` 之前**；**`close()` 放 `finally`**；
`TaskArgs` 的顺序必须与 callable 的 `signature` 位置对应。

### 6.2 `@scene_test` 的契约

```python
@scene_test(level=2, runtime="tensormap_and_ringbuffer")   # 只有这两个参数
class TestMyExample(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/my_orch.cpp",   # 必填，相对测试文件
            "function_name": "aicpu_orchestration_entry",     # 必填，须匹配导出符号
            "signature": [D.IN, D.IN, D.OUT],                 # 选填
        },
        "incores": [                                          # 必填（可为 []）
            {"func_id": 0, "source": "kernels/aiv/k.cpp", "core_type": "aiv"},
        ],
    }
    CASES = [
        {"name": "default", "platforms": ["a2a3sim", "a2a3"], "params": {}},
    ]
    # 选填：RTOL / ATOL（默认 1e-5）、SKIP_GOLDEN、RUNTIME_ENV

    def generate_args(self, params):      # 必须实现
        return TaskArgsBuilder(TensorArg("a", torch...), Scalar("n", 128))

    def compute_golden(self, args, params):   # skip_golden 时可省
        args.f[:] = ...                        # 原地写输出

    # def compare_outputs(self, test_args, golden_args, output_names, params): ...  # 选填

if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
```

**两个最常见的"报错很怪"的原因**：

- `function_name` 与 orchestration 里 `__attribute__((visibility("default")))` 导出的符号不一致
- `func_id` 与 orchestration 里 `rt_submit_aiv_task(0, ...)` 传的 id 不一致

**平台在 `CASES` 里声明，不在装饰器上。** 装饰器没有 `platforms` 参数，传了是 `TypeError`。

### 6.3 运行方式

```bash
# pytest（sim，无需真实设备）
pytest examples tests/st --platform a2a3sim --device 0-3 --manual include
pytest examples/my_example --platform a2a3sim

# 单文件独立运行
python examples/my_example/test_my_example.py -p a2a3sim --manual include

# 真硬件
pytest examples tests/st -m "not sdma" --platform a2a3 --device 4-7

# 单测
pytest tests/ut -m "not requires_hardware"
cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure
```

> 被标 `manual` 的用例默认跳过，需要 `--manual include`。

### 6.4 pytest 参数

| 参数 | 取值 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--platform` | `a2a3` / `a2a3sim` / `a5` / `a5sim` | — | 无 argparse 校验，拼错会在后面才报 |
| `--device` | `0` / `0-7` / `0,2,5` | `0` | 设备 id 或设备池。**sim 下也要给足** —— L3 用例声明了 `device_count`，池子不够会整体中止并提示 `needs N devices but pool has M` |
| `--case` | 可重复；`Foo` / `Cls::Foo` / `Cls::` | — | 用例选择 |
| `--manual` | `exclude` / `include` / `only` | `exclude` | manual 用例处理 |
| `--runtime` | 两个 runtime 名之一 | — | 限定运行时 |
| `--level` / `--exclude-level` | `2` / `3` / `4` | — | 限定/排除层级 |
| `--max-parallel` | `auto` 或整数 | `auto` | **CPU 少时务必调小**，见 §6.7 |
| `--rounds` | 整数 | `1` | **> 1 会强制关闭所有诊断开关** |
| `--skip-golden` | 标志 | 否 | 基准测试模式 |
| `--sanitizer` | `asan`/`ubsan`/`tsan`，或裸 `-fsanitize` token | `none` | 须与运行时构建时的一致 |
| `--pto-session-timeout` | 秒，`0`=关 | `0` | 超时退出码 `124` |

诊断开关（详见 §7）：`--enable-chip-swimlane [1-4]`、`--dump-args {off,partial,hybrid,full}`、
`--enable-pmu`、`--enable-dep-gen`、`--enable-scope-stats`、`--enable-swimlane-overhead`。

> 独立运行（`run_module`）的参数名基本一致，但 **`--dump-args` 在那里取整数**（裸标志 = 1），
> `--level` 只接受 `[2, 3]`。

### 6.5 `CallConfig`：运行期可调项

```python
config = CallConfig()
config.aicpu_thread_num = 0          # 0 = 自动（a2a3 → 4，a5 → 5）；显式值须 >= 2
config.runtime_env.ring_task_window = 16384          # 标量 → 广播到 4 个 ring
config.runtime_env.ring_heap = 256 * 1024 * 1024
config.output_prefix = "outputs/mine"  # 开任何诊断都必须设，否则 validate() 抛异常
```

`runtime_env` 是嵌在 `CallConfig` 里的对象（不是 dict），三个字段各是 4 元素数组
（对应 4 个 ring），赋标量广播、赋 4 元素列表逐 ring 设置：

| 字段 | 约束 | TRB 默认 | HBG |
| --- | --- | --- | --- |
| `ring_task_window` | 非零时须是 `[4, INT32_MAX]` 的 **2 的幂** | 16384 | 只读 `[0]`；其内部解析只要求正数，但上面这条校验在 `CallConfig::validate()` 里，对所有运行时一视同仁 |
| `ring_heap` | 非零时须 **≥ 1024**（字节/ring） | 256 MiB ×4 | **忽略**（会打 warning） |
| `ring_dep_pool` | 非零时须在 `[4, INT32_MAX]` | 16384 | 不读 |

**没有 `block_dim`** —— 一次 run 独占整个设备。

在 `@scene_test` 里通过每个 case 的 `config` 字典设置，合法键恰好五个：
`aicpu_thread_num` / `runtime_env` / `device_count` / `num_sub_workers` / `launch_depth`。
**未知键在类导入时就抛 `ValueError`。**

> **命名陷阱**：类属性 `RUNTIME_ENV`（设置 OS 环境变量）与每个 case 的
> `config["runtime_env"]`（ring 尺寸）是完全无关的两套机制。

### 6.6 容量上限与失败模式

**TRB** —— 环形缓冲被撑爆时会锁存明确的错误码（`src/common/runtime_status/error_names.h`）：

| 码 | 名称 | 含义 |
| --- | --- | --- |
| 1 | `SCOPE_DEADLOCK` | 一个 scope 内任务数撑满 ring；fanout 引用要到 `scope_end` 才释放 |
| 2 | `HEAP_RING_DEADLOCK` | 分配器无法为新任务预留堆空间 |
| 3 | `FLOW_CONTROL_DEADLOCK` | 任务表无可用槽位 |
| 4 | `FANIN_CAPACITY_EXCEEDED` | 内联 fanin（`CHIP_FANIN_INLINE_CAP = 64`）放不下后，承接溢出的 dep pool 也耗尽；用 `ring_dep_pool` 调大 |

这些都由 `tests/st/runtime_fatal_codes/` 端到端验证。

**HBG** —— 全图必须装得下，无回收：

- 默认任务表 16384，可经 `ring_task_window[0]` 调大，硬上限 `1 << 20`
- 更早触发的实际上限：共享内存镜像须 < `INT32_MAX` 字节（32 位自相对偏移）
- 超限 → `FLOW_CONTROL_DEADLOCK`；堆耗尽 → `HEAP_RING_DEADLOCK`（**堆没有调节旋钮**，
  只能缩小图的中间张量）

**一个硬约束（onboard）**：*一个 device 在一个进程里只能跑一种 runtime*。
CANN 的 `libaicpu_extend_kernels.so` 用单例缓存 AICPU `.so`，第二个 runtime 会调到
第一个的函数指针并**挂死**（见 [`docs/testing.md`](../testing.md) 的 Runtime Isolation Constraint）。
pytest 靠"每 runtime 一个子进程"规避；
自己驱动 `Worker` 时切勿在一个进程内切换 runtime。

### 6.7 sim 的性能数字没有参考价值

sim 把每个 AICore 变成一个主机线程 —— a2a3 是 72 个，a5 是 108 个。
在 8 核机器上这是严重过订阅，会导致活锁或误超时
（见 [`docs/troubleshooting/sim-oversubscription-hang.md`](../troubleshooting/sim-oversubscription-hang.md)）。

- **缓解**：`--max-parallel 2`
- **不要**把 sim 的耗时当作硬件性能的任何指示

---

## 7. 可观测性（DFX）

这个仓库在诊断上投入很重 —— 因为片上故障信号本身就是模糊的。

| 开关 | 运行时产物 | 后处理产物 |
| --- | --- | --- |
| `--enable-chip-swimlane` | `chip_swimlane_records.json` | `merged_swimlane.json`（Perfetto 格式） |
| `--enable-pmu` | `pmu.csv` | — |
| `--enable-dep-gen` | `deps.json` | `deps_viewer.txt` |
| `--enable-scope-stats` | `scope_stats/scope_stats.jsonl` | `scope_stats/scope_stats.html` |
| `--dump-args` | `args_dump/args_dump.json` + `args.bin` | 用 `dump_viewer` 看 |

产物落在 `outputs/<类名>_<用例名>_<时间戳>/`；L3 的 fork 子进程再深一层 `rank<N>/d<M>/`。
**只有至少开启一个诊断时才会创建该目录**，且后处理在 `finally` 里执行 —— 用例失败也有产物。

分析 CLI（全部是 `python -m simpler_setup.tools.<name>`）：

| 工具 | 用途 |
| --- | --- |
| `swimlane_converter` | 性能 JSON → Perfetto 轨迹 |
| `strace_timing` | 分阶段/分轮次的 `[STRACE]` 主机计时表 |
| `critical_path` | 关键路径分析 |
| `sched_overhead_analysis` | 调度开销模型 |
| `deps_viewer` | 依赖图 → 文本或可缩放 HTML |
| `core_swimlane` | AICore 核内重放（需 `msprof op simulator`） |
| `dump_viewer` | 查看 args 转储 |

**先用这些自带工具，不要手写插桩** —— `.claude/skills/dfx-analyze/` 有完整流程。

每次运行还会输出 `[STRACE]` 主机跨度，形如：

```text
chip.run                      dur=933ms
├─ chip.run.bind              dur=65µs
├─ chip.run.publish_image     dur=27µs
├─ chip.run.prepare_execution dur=839ms
├─ chip.run.runner_run        dur=94ms
│  └─ device_wall             dur=93ms  clk=dev
└─ chip.run.validate          dur=3µs
```

---

## 8. 加一个新示例

**没有注册表要改** —— pytest 整体收集 `examples/` 和 `tests/st/`，目录布局纯属约定。

```text
examples/<arch>/<runtime>/my_example/
├── test_my_example.py          # @scene_test 类
└── kernels/
    ├── orchestration/my_orch.cpp   # 必有
    ├── aiv/*.cpp                   # 可选
    └── aic/*.cpp                   # 可选
```

最快的起点是拷贝
[`examples/a2a3/tensormap_and_ringbuffer/vector_example/`](../../examples/a2a3/tensormap_and_ringbuffer/vector_example/)
—— 40 行的类，3 个 incore，2 个 case。

流程：

1. 建目录与 `kernels/` 子目录
2. 写 orchestration `.cpp`（用 `rt_submit_aiv_task` / `rt_submit_aic_task` 提交任务，
   依赖靠 TensorMap 从张量自动推断，`SIMPLER_SCOPE()` 管中间张量生命周期）
3. 写 kernel `.cpp`（`extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args)`）
4. 写 `test_*.py`（见 §6.2）
5. `pytest <dir> --platform a2a3sim` → `python <dir>/test_*.py -p a2a3sim`

更详细的教程见 [`docs/user/how-to/write-and-run-a-kernel.md`](../user/how-to/write-and-run-a-kernel.md)。

---

## 9. 开发前必读

| 文件 | 为什么 |
| --- | --- |
| [`.claude/rules/codestyle.md`](../../.claude/rules/codestyle.md) | 13 条，含"分发路径上不准 sleep"、命名规范、遗留名退役策略 |
| [`.claude/rules/comments.md`](../../.claude/rules/comments.md) | 注释只写"现在是什么"，不写"为什么改" |
| [`.claude/rules/doc-consistency.md`](../../.claude/rules/doc-consistency.md) | 改代码要在同一个 commit 里改文档 |
| [`docs/investigations/`](../investigations/README.md) | **提优化前先 grep** |
| [`docs/capability-survey.md`](../capability-survey.md) | 哪些能力已交付/被门控/仅设计 |

改 TRB 时记得**两个架构的树要同步改**（近似拷贝，见 §4.2）。

### 测试策略

| 层 | 跑什么 | 需要什么 |
| --- | --- | --- |
| C++ 单测 | 调度算法（四类调度器都有覆盖）、TensorMap、Ring、Scope | 任何机器 |
| Python 单测 | 框架、编译链、DFX 工具 | 任何机器 |
| 场景测试 + sim | 端到端功能、计算正确性 | g++-15 |
| 场景测试 + onboard | 真实时序、性能 | 昇腾硬件 |

C++ 单测能覆盖调度逻辑是因为**调度器源码是可移植 C++**：生产时编到 AICPU（或 a5 的 AIV），
测试时编到主机，只把碰硬件的原语换成桩（`tests/ut/cpp/support/`：寄存器、缓存维护、设备时钟、设备日志）。
调度逻辑本身原样编译。

---

## 10. 延伸阅读

| 主题 | 文档 |
| --- | --- |
| **调度器详解**（要改/写调度逻辑先读这个） | [zh-cn/schedulers.md](schedulers.md) |
| 任务类型全景与用例统计 | [zh-cn/task_types.md](task_types.md) |
| 文档总索引 | [docs/README.md](../README.md) |
| 使用者入口 | [docs/user/](../user/README.md) |
| L2 芯片架构 | [docs/chip-level-arch.md](../chip-level-arch.md) |
| L0–L6 层级模型 | [docs/hierarchical-level-runtime.md](../hierarchical-level-runtime.md) |
| 任务数据流 | [docs/task-flow.md](../task-flow.md) |
| Orchestrator 内部 | [docs/orchestrator.md](../orchestrator.md) |
| Scheduler 内部 | [docs/scheduler.md](../scheduler.md) |
| 硬件底座 | [docs/hardware/](../hardware/README.md) |
| DFX 全索引 | [docs/dfx/](../dfx/README.md) |
| 故障排查 | [docs/troubleshooting/](../troubleshooting/README.md) |
| a2a3/a5 运行时差异 | [docs/tensormap-and-ringbuffer-a2a3-vs-a5.md](../tensormap-and-ringbuffer-a2a3-vs-a5.md) |
