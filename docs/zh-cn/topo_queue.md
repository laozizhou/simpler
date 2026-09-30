# topo_queue 开发总结

a5 上的一个新调度方案原型：**AICPU 对已就绪的 DAG 做一次性排序、写成 GM 任务表后退出调度；
AICore 空闲时自取任务，靠每任务一个的完成 counter 自行解依赖**。

当前状态一句话：**仿真端到端已验证**（bgemm 500 任务、AIC+AIV 混合、torch 对答案、重复稳定），
**真机未跑**，已知欠账见第 5 节。

---

## 1. 整体思路

### 执行模型

```text
AICPU（一次性）                      AICore × N（常驻，互不通信）
─────────────────                   ──────────────────────────────
读 HBG 已上传的图                     loop:
  ↓                                   idx = peek(queue_head)
校验 + 生成 order[]                    if idx >= task_count: 收工
  ↓                                   task = order[idx]
装好每任务的 DispatchPayload           if 类型不匹配本核: 不取，闲一拍     ← 验类型
  ↓                                   if CAS(head, idx, idx+1) 失败: 重看  ← 认领
连同 counter 区打包成 image 写入 GM     for p in 前驱: 阻塞等 counter[p]==DONE
  ↓                                   执行 kernel（拆 payload）
发布地址、开寄存器窗口                  counter[task] = DONE
  ↓
只盯 error/retired 两个字 → 敲退出铃 → return（= host 眼里的运行结束）
```

没有 ready queue、没有每任务的派发寄存器写、没有回收。counter 三态：0=PENDING、1=DONE、2=FAILED。

### 为什么不死锁（要点）

- 队列是拓扑序（前驱 id < 自己，图校验强制），head 只被**匹配类型的核 CAS 认领**才前进，绝不跳过；
- 已认领未完成的任务中，全局下标最小的那个前驱必然全部终结，永远可跑；
- 队头是 T 类型时：有空闲 T 核就被认领；T 核全忙时，最小下标任务终会跑完、其持有核回到队头。

三个不变量，破坏任何一个都要自带新证明：拓扑序、head 只经认领前进（**跳过=死锁**：被消费的下标没人执行）、已认领任务必达终态（每条退出路径都发布 DONE/FAILED）。完整论证在 `topo_worker.h` 头注释。

### 与既有调度器的关系

topo_queue 属于 HBG 模型（图提前建好、整图常驻），是该模型下继「AICPU 推派发」（legacy）、
「常驻 AIV 集中调度」（resident）之后的第三种执行策略：**去中心化自取**。
它复用 HBG 的图（建图/依赖推导/上传一行未写），只替换「照图执行」这一段——
resident 的 ~10000 行 scheduler/ 机器被 ~470 行换掉，代价是把等待搬到每个工人核上自旋。

---

## 2. 关键设计决策

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 单队列 vs 双队列 | **单队列 + 抓取时验类型**（peek+CAS） | 简单可靠；接受队头类型空转（HOL）作为性能税。双队列版在 git 历史里，worker 按 (order,count,head) 参数化，切换是参数级替换 |
| 排序 | **恒等序 + 无条件验证** | HBG task id 天生拓扑序（图校验强制 producer<id）。order[] 数组保留为将来 est/层序排序的插槽——换策略时 device 侧零改动 |
| 任务表存什么 | **现成的 DispatchPayload**（方案 A） | AICPU 复用 HBG 的 materialize 装箱：args/context 槽位约定继承而非复刻，错位风险归零；aicore 执行段与 legacy 同构 |
| host 侧 | **复用 HBG 源码**（build_config 指向其目录） | 省 ~5000 行样板；耦合面收敛到图的 wire 格式与 Runtime/Handshake（aicore 本就在用） |
| GM 原语 | **自有 topo_gm_memory.h**，只依赖平台层 | 隔离活跃开发中的 HBG scheduler 头；并躲开一个真实缺陷：经 HBG 头链会把 `SPIN_WAIT_HINT` 静默变成空操作（KNOWN_ISSUES 有档） |
| kernel 地址来源 | host 的 **callable 表**按 kernel_id 直取 | `SCHEDULER_CORE_CALLABLE_RESOLVED_ADDR_OFFSET=136` 是 HBG 里无人使用的死常量（且落在 payload 区内），勿用 |
| 失败处理 | counter 加 FAILED 态 + run 级错误锁存（先锁根因再发布）+ 10s 等待兜底 | `RunControl::error` 是权威判决；counter 只回答「这个任务终结了没」。没人碰到的任务停在 PENDING |

---

## 3. 程序框架

### 文件清单（src/a5/runtime/topo_queue/）

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `runtime/topo_worker.h` | ~310 | **心脏**：peek→验类型→CAS→等前驱→执行→发布；死锁论证；模板化于 Platform policy（同一份源码被硅片和 host 单测驱动） |
| `runtime/topo_prepare.h` | ~206 | **AICPU 的大脑**（esl_proxy 七步骨架落地）：读图→classify 体检→恒等序→抄 fanin 进 CSR→查 callable 表→materialize 装 payload→拓扑性复查 |
| `aicpu/aicpu_executor.cpp` | ~301 | 手脚：多线程进入选主（CAS）、收报到（epoch 验收+rmb）、建 image、flush 后发布+开窗、盯 error/retired、敲铃收尾。**失败纪律：任何失败路径都先放行核（空 image+开窗）再返回，否则挂死** |
| `aicore/aicore_executor.cpp` | ~215 | 核侧外壳：五阶段握手（借 legacy 协议）+ 拆 payload 执行；绑定时校验 payload_stride 防两侧版本错配 |
| `runtime/topo_image.h` | ~257 | image 布局合同：只存偏移不存指针；header alignas(64)；layout 只有一份、写读共用；uint32 溢出即拒绝 |
| `runtime/topo_gm_memory.h` | ~214 | GM 原语（query/store/fetch_add/CAS/observe/publish/spin_hint），CCE 与 sim 双实现 |
| `runtime/topo_platform_device.h` | ~190 | AICore 侧 Platform policy，每个方法注明所欠内存序 |
| `runtime/topo_queue_types.h` | ~154 | wire 类型：counter/队头/RunControl 各独占 cache line（整线回写防互踩）；64 位——device 原子仅此宽度 |
| `runtime/topo_sort.h` | ~164 | Kahn 排序：生产路径暂不用（恒等序），供测试构造合法序 + est 插槽 |
| `build_config.py(.parked)` | 50 | host/orchestration 指向 HBG 目录；aicpu/aicore 为本 runtime 自有 |

### image 内存图（一整块连续 GM，一次分配一次清零）

```text
header(128B, 偏移表+magic+version) → order[] → fanin CSR → entries[](id+类型,8B)
→ payloads[](每任务一个 DispatchPayload,64 对齐) ── 以上发布后只读 ──
→ counters[](每任务 64B 独占一线) → queue_head(64B) → run_control(error+retired 各一线)
```

按访问模式列式拆分：验类型只碰 8B entries；counter 多核写共享必须独占线；只读区开机一次 observe 覆盖。
清零即合法初态（PENDING=0、head=0、error=OK）。`counter_of(id) ≡ counters[id]`——映射是下标，不是数据。

### 关键协议约定

- 发布 = 写货 → 屏障 → 立标记；消费 = 见标记 → 屏障 → 读货。counter 的 DONE 前有整 DCache 写回（kernel 标量输出）。
- 一切跨核只读大块（header 两条线、只读区）按**尺寸** invalidate，不按单线。
- 错误锁存 CAS「第一个写者赢」；执行失败**先锁根因再发布 FAILED**（反序会被消费者的「果」抢走「因」）。

---

## 4. 仿真测试

### 四层验证

**单元（8 目标，全绿）**：sort（畸形输入）、worker（并发正确性）、type_gate（门的四条性质＋96 任务跨类型链×8 线程×30 轮「每任务恰好一次且跑在自己类型的核上」）、failures（五条失败分支逐条＋根因抢锁——竞态窗口由 policy 按住，确定性复现）、stress（随机 DAG 模糊，带死线永不挂死）、image（布局不变量）、prepare（字节级伪造 HBG 图驱动建表全链路，含「materialize 产物逐位符合 HBG args 约定」）、platform_contract（签名/宽度/布局断言）。

**变异（10/10 被杀）**：逐条删改 worker 的关键分支必须变红。含三个门专属变异：删类型门→跑错核、外类型跳过→任务失联、无视 CAS→跑两遍。教训：竞态类修复的测试要靠 policy 接缝**按住窗口**（yield 100 次），撞运气 200 次也撞不到。

**端到端**：HBG 基线通过→topo_queue bgemm 通过→重复 5/5→**证伪测试**（藏起本 runtime 的 aicpu.so 必须失败——绿灯不证明跑的是谁，按预期变红才证明）。

**回归**：动过的共享文件（kernel_compiler 两处映射）后，HBG bgemm / TMR vector_example / 全仓库 UT 259/259 均绿。

### sim 性能对比（仅作健全性信号，禁止外推）

同负载 bgemm 500 任务 device_wall：HBG resident ~2.6–2.7s（2 样本）；topo_queue **0.86–0.90s**（7 样本，稳定）。
**不可外推到硅片**：sim 里 dcci=全内存栅栏（重罚 HBG 的密集 observe）、无真实 GM 延迟（轻放 topo_queue 的轮询）、24 线程超订在 ~10 host 核上、24 核规模压不出 108 核的队头争抢与 HOL。

### sim 证明了 / 证明不了

| 证明了（终局） | 证明不了（真机专属） |
| --- | --- |
| 调度逻辑正确（依赖序、恰好一次、golden 一致） | 内存序（sim cache 是空模型，`SINGLE_CACHE_LINE==0`） |
| 协议完备（握手咬合、无挂死、干净收尾） | 地址空间（`__gm__` 在 sim 为空宏，未过 ccec） |
| 工程可运行（真实管线、三家共存、host 复用） | 硅片性能、108 核规模行为、HOL 实际代价 |

备注：TSan 直查真 .so 在 macOS 上不可行（gcc-15 无 arm64 libtsan；tsan dylib 被系统策略拒载入未插桩 python），
未做；真机前非必需。UT 层 TSan（测试替身路径）已由压测覆盖、无竞争报告。

---

## 5. 真机测试建议

### 欠账清单（上机前后必还）

| # | 欠账 | 性质 | 状态 |
| --- | --- | --- | --- |
| 1 | **`__gm__` 地址空间链**：ccec 把地址空间和执行位置都编进类型，sim 下两者皆为空宏所以编得过 | **编译必败** | **已还**。`TOPO_GM` 穿透 worker loop 与 platform policy；lambda 的 `operator()` 是 host 函数且不接受注解，改为带 `__aicore__ operator()` 的 `TaskRunner`；两个 loop 模板加 `__aicore__`。宏在 host 构建为空，单测不受影响 |
| 2 | **GM 分配接缝**：真机上 AICPU 堆 ≠ AICore 可见 GM | 编译能过、**首跑必错**（核读到垃圾） | **已改，真机未验**。`topo_image_allocate()` 改用平台的 `aicpu_device_malloc()`（onboard 经 halMemAlloc 拿 HBM 设备虚址，sim 仍是 malloc），自行向上取整到 cache line 并保留原始指针供 free。该 API 此前全仓库无调用者，halMemAlloc 路径尚未被任何运行验证；若失败，退路是复用 HBG 的 `scheduler_state_base_address`（topo_queue 不用它）。**约束仍在：payload 里有 materialize 写入的绝对自指针，image 必须在最终 GM 地址上原地填充——先分配后填充，绝不能填好再 memcpy** |
| 3 | 内存序 | 只有硅片能审 | 见阶梯 3 |
| — | 小项：`COMPLETION_TIMEOUT=60s` 按图大小调；双架构 UT 门不拦运行、只拦 `tests/ut/cpp` 的 cmake | | aicpu 的 aarch64 交叉编译已验证通过 |

### 阶梯（每级一条命令一个判据）

```bash
# 阶梯 1（纯编译判决——欠账 1 与交叉编译在此一并暴露）
command -v ccec && ls $ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++
mv src/a5/runtime/topo_queue/build_config.py.parked src/a5/runtime/topo_queue/build_config.py
python -m pip install --no-build-isolation --config-settings=build.targets=build_package_a5 -e . 2>&1 | tee build/logs/onboard_build.log
# 判据：错误清单归零。编译不碰 NPU，但共享机上仍走队列：task-submit --no-device --run "..."。

# 阶梯 2（占卡 1 次，首跑裁决欠账 2）
.claude/skills/onboard-arch-precheck/check.sh a5 || exit 1
mkdir -p out/ascend && export ASCEND_PROCESS_LOG_PATH=$PWD/out/ascend
task-submit --timeout 1800 --max-time 1800 --device auto --device-num 1 \
  --run "python examples/a5/topo_queue/benchmark_bgemm/test_benchmark_bgemm.py -p a5 --manual include -d \$TASK_DEVICE"
# 判据：PASSED；失败看 out/ascend/device-*/device-*.log 里本 runtime 的 LOG_ERROR（错误码+任务号）

# 阶梯 3（占卡持锁，内存序审判：一次通过不算数，判据是重复稳定）
task-submit --timeout 7200 --max-time 7200 --device auto --device-num 1 \
  --run "for i in \$(seq 1 50); do python ...bgemm... -p a5 --manual include -d \$TASK_DEVICE || exit 1; done"
```

### 症状分诊（真机独有的错法）

| 症状 | 第一怀疑 |
| --- | --- |
| 核读到 image 全是垃圾 / 立刻乱 | 欠账 2：malloc 不是 GM |
| 结果偶发错、counter 已 DONE 但数据旧 | 发布序：DONE 前的整 DCache 写回缺失/失效 |
| 随机挂死在等前驱（超时报 TOPO_ERR_WAIT_TIMEOUT=3） | 消费序或读前 invalidate；也可能真是前驱核死了 |
| 全体核不开工 | 握手：epoch 验收 / 窗口未开 / image 地址未 flush |
| TOPO_ERR_BAD_IMAGE=6 | aicpu 与 aicore 二进制版本错配（payload_stride 校验拦下的） |

规模提醒：108 核下重点观测**队头 CAS 争抢**与**类型成簇负载的 AIV 空转**（HOL 账单）——bgemm 类型交错友好，
qwen 类按层聚簇的图才是试金石（但 SPMD 被 v1 闸门拦着，届时走 HBG 对照即可）。

---

## 6. 使用方法

```bash
# 开关（因 UT 的双架构对称门，a5-only runtime 会让 tests/ut/cpp 的 cmake 报错，故平时停牌）
# 开：mv src/a5/runtime/topo_queue/build_config.py.parked src/a5/runtime/topo_queue/build_config.py
# 关：mv src/a5/runtime/topo_queue/build_config.py src/a5/runtime/topo_queue/build_config.py.parked
# （python 与 cmake 两套发现都只认精确文件名 build_config.py，一次改名同时控制两边）

# 构建三件套（开关 on 时 pip install 自动含它；或单独）
python -c "from simpler_setup.runtime_builder import RuntimeBuilder; \
  RuntimeBuilder(platform='a5sim').get_binaries('topo_queue', build=True)"

# 仿真运行（macOS 注意 PATH 带上 homebrew、单线程 OMP）
export PATH="/opt/homebrew/bin:$PATH" OMP_NUM_THREADS=1
python examples/a5/topo_queue/benchmark_bgemm/test_benchmark_bgemm.py -p a5sim --manual include

# 单元测试
ctest --test-dir tests/ut/cpp/build -R topo_queue
```

改动过的共享文件（仅 2 处）：`simpler_setup/kernel_compiler.py`（两行映射：topo_queue 的编排工具链与
头文件契约 = HBG 的）、`tests/ut/cpp/a5/runtime/CMakeLists.txt`（一行 add_subdirectory）。
注意：开关 off 时跑 `pytest examples` 会在 topo_queue 例子上报错（runtime 未发现）。

---

## 7. 转正前清单（进主干才需要）

1. 双架构门：a2a3 薄 twin（桩 aicpu/aicore + `src/common/topo_queue/` 转发头 + UT 镜像目录，半天到一天）**或**向维护者提议放宽按架构声明；
2. ccec 首编验证扩到 a2a3；
3. `kernel_compiler.py` 的两处名字映射建议上游化为「由 build_config 声明契约」（维护者的设计决定）;
4. GM 分配接缝正式方案（欠账 2 的定稿）；
5. est/层序排序接入 order[] 插槽（等真机 profiling 数据回填权重；esl_proxy 的 clc_prepare.c 为参照，含 w>0→免费拓扑序的论证与确定性 tie-break）。
