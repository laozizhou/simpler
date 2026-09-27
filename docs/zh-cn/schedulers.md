# simpler 的调度器全景

写给要**修改现有调度逻辑或新增一个调度器**的人。仓库总览见 [overview.md](overview.md)。

**先说三条最重要的结论：**

1. **没有统一的调度器抽象** —— 没有基类、没有虚表、没有 `ops` 结构体、没有 concept。每套调度器都是独立实现。
2. **实际有 6 个实现，不是 4 个**（见 §1 的表；多出来的两个是兄弟架构的克隆，见 §4）。
3. **跨调度器的复用只有两种**：`src/common/host_build_graph/` 里真正共享的头文件，以及**兄弟架构目录之间的拷贝血缘**。

---

## 1. 六个调度实现的分布

| # | 名称 | 位置 | 说明 |
| --- | --- | --- | --- |
| a | **L3+ 主机调度器** | `src/common/hierarchical/scheduler.{h,cpp}` | 唯一被 L3 以上使用 |
| b | **L2 TRB AICPU** | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/` | |
| b' | a5 TRB AICPU | `src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/` | 与 b 近似克隆 |
| c | **L2 HBG AICPU（a2a3）** | `src/a2a3/runtime/host_build_graph/runtime/scheduler/` | |
| c' | a5 HBG **legacy** AICPU | `src/a5/runtime/host_build_graph/runtime/scheduler/*.cpp` | c 的克隆，作为 d 的回退 |
| d | **a5 HBG 常驻 AIV** | `src/a5/.../host_build_graph/aicore/aicore_executor.cpp` + `runtime/scheduler/scheduler_{ready,dispatch,completion,…}.h` | 跑在 AICore 上 |

> **c' 与 d 住在同一个目录里**：4 个 `.cpp` 加 `scheduler.h` / `scheduler_context.h` 属于 AICPU 调度器；
> `scheduler_{ready,dispatch,completion,memory,graph,ssbuf,watchdog,types}.h` 才是 AICore 常驻调度库；
> `scheduler_layout.h` / `scheduler_topology.h` 两边共用。
> 源码注释在 `scheduler_dispatch.h:14` 明确区分了两者，改代码时别混淆。

---

## 2. 对比总表

| | (a) L3 主机 | (b) L2 TRB AICPU | (c) L2 HBG AICPU | (d) a5 常驻 AIV |
| --- | --- | --- | --- | --- |
| **执行位置** | 主机 CPU，`std::thread` | 设备 AICPU | 设备 AICPU | **AICore（AIV）** |
| **线程/核数** | 每个 Worker **1 个**调度线程 | `aicpu_thread_num - 1` 个 S 线程（+1 orchestrator） | **3S+1P**：`N-1` 个 S + **1 个专职 resolution 线程 P** | **每 cluster 1 个 Scheduler AIV**，另 2 个 lane 是纯 Executor |
| **核归属** | worker id 在 `init` 时冻结 | cluster 轮转 `ci % threads` | 同 b | 固定：Scheduler 拥有自己那 3 个 lane |
| **就绪队列** | mutex FIFO，按 `RunId` 分区 | 无锁 MPMC（Vyukov） | 同 b + `graph_prepare_queue` + SPSC `sp_queues_` | GM 侵入式链表 + 分片位图目录 |
| **分区维度** | `RunId` → worker-id / group / SUB | `ResourceShape`(AIC/AIV/MIX) × tier | 同 b | `core_type`(AIC/AIV) × owner scheduler |
| **ready 机制** | fanin 计数器 + CAS | fanin 原子refcount + 一次性 claim | **轮询状态字节 + 侵入式唤醒链表** | **轮询状态 + GM 唤醒链表** |
| **派发策略** | run-head 优先；group 全有或全无 | shape 优先级 + IDLE>PENDING | 同 b + 有界图物化 | 最小负载优先 + 分片内窃取 |
| **完成观测** | 轮询 mailbox + 条件变量 | 轮询 **MMIO COND 寄存器** | 同 b，但结果交给 P 线程 | **SSBUF 中的 generation 字** |
| **回收** | Ring 释放 | CONSUMED → 推进 ring | **无**（全图常驻） | **无**（全图常驻） |

---

## 3. 逐个详解

### (a) L3+ 主机调度器

**执行位置** —— 每个 `Worker`（每个层级）一个专属 OS 线程，跑在主机 CPU 上。
`std::thread sched_thread_`（`scheduler.h:159`），在 `Scheduler::start`（`scheduler.cpp:122`）启动。
**只有一个调度线程**；任务在别的进程（fork 出的子进程）里执行，调度器从不执行用户代码。

**主循环** —— `Scheduler::run()`，`scheduler.cpp:257-359`：

```text
while (true) {
    // 阶段 0：空闲等待 —— 仅当没有任何 endpoint lane 忙碌时才进入
    if (!manager->any_busy())  completion_cv_.wait(...)        // :273-279
    manager->progress();          // 轮询每条 endpoint lane     // :298
    // 阶段 1：排空 completion FIFO
    while (pop completion_queue_) on_task_complete(c);         // :302-312
    // 阶段 2：派发
    if (!stop_requested_) dispatch_ready();                    // :320
    // 阶段 3：终止
    if (stop_requested_ && !any_busy()) { final drain; break; }// :342-357
}
```

`dispatch_ready()`（`:473-505`）内部顺序是固定的：

```text
1. active_run_cb() == INVALID_RUN_ID → 直接返回      ← 运行准入闸门
2. activate_prepared_run(run)
3. dispatch_preparable_next_level_singles()          ← 预备/暂存派发
4. authorize_staged_launch(...)                      ← 早期启动授权
5. do { group → singles } while (retry)              ← group 优先于 single
6. dispatch_sub_ready()                              ← SUB 最后
```

**任务如何变 ready** —— **计数器 + 互斥锁下的 CAS**。
`TaskSlotState::fanin_count` / `fanin_released`（`types.h:356-357`）。
生产者完成时 `fanin_released.fetch_add(1)`（`scheduler.cpp:396`），
`try_mark_ready(cs)`（`types.cpp:61-67`）在 `fanout_mu` 保护下比较并翻转状态 —— **只有一个调用者能赢**。
赢家调用 `enqueue_ready_cb`，绑定到 `Orchestrator::enqueue_ready`（`orchestrator.cpp:1151-1175`）—— 这是**唯一的 READY 路由点**。

**关键判断点**（写新策略时最需要看的部分）：

| 场景 | 位置 | 条件 |
| --- | --- | --- |
| 运行准入 | `:477-479` | `active_run_cb() == INVALID_RUN_ID` → 什么都不派发 |
| 暂存派发 | `:530-532` | 有 group 排队 或 该 run 已暂存 → 不暂存 |
| | `:537` | `!worker->can_stage()` → 跳过（注意与 `idle()` 不同） |
| | `:552` | `claim_for_dispatch(state)` —— READY→RUNNING 的原子 CAS |
| **Group 全有或全无** | `:659-665` | **每个成员的 worker 都必须 `idle()`** |
| | `:666-682` | **任一忙碌 → 整组不派发，且预留所有目标 worker** |
| Single 派发 | `:754` | 被 group 预留的 worker 直接跳过 |
| | `:773-779` | **group 屏障**：该 run 还有 group 排队 → 退回并重跑整趟 |
| | `:786` | `break` —— **每个 worker 每趟最多派一个任务** |
| SUB 派发 | `:574-584` | 需要 `group_size()` 个**不同的**空闲 SUB worker，凑不齐就整趟停止 |

**背压**：没有队列容量检查。背压来自运行准入、`PipelineSlotLease` 和 `WorkerEndpointCaps::max_inflight_tasks`。

**完成观测** —— `manager->progress()` 每轮轮询所有 endpoint lane，底层是 `MAP_SHARED` mailbox 轮询 `TASK_DONE`。
完成链：`poll_progress` → `worker_done`（**group 聚合**：只有 `group_terminal_count == group_size` 时才推一个聚合完成）→ `completion_queue_` → `on_task_complete` → 消费者 fanin 释放 → `try_consume` → `on_consumed_cb`（TensorMap 擦除 + Ring 释放）。

唤醒模型是**边沿触发**的，契约写在 `docs/scheduler.md:61-81`：任何让排队工作变得可放置的状态变化，都必须推一个 completion 或 bump `wake_generation_`。安全阀是 `:273` 的 `if (!any_busy())` —— 只要有 lane 忙，循环就不休眠。

**扩展点** —— `Scheduler::Config`（`scheduler.h:87-122`）是真正的接缝：8 个 `std::function` 钩子
（其中 `before_claim_cb` / `after_group_phase_cb` 是测试接缝，生产不设），其余 6 个在 `worker.cpp:189-214` 接线。
`class WorkerEndpoint`（`worker_manager.h:401-478`）是这层主要的虚接口 —— 但它抽象的是**传输**
（`WorkerEndpointKind`：`LOCAL_MAILBOX` / `REMOTE_L3` / `MPI_GROUP_MAILBOX`），**不是策略**。
（同目录还有 `RemoteL3Transport`，`remote_endpoint.h:29-41`。）
**放置策略本身硬编码在 `Scheduler` 的方法里，没有策略对象。**

---

### (b) L2 TRB AICPU 调度器

**执行位置** —— 设备 AICPU，N 个线程。**最后一个是 orchestrator，其余是调度线程**：

```cpp
aicpu_thread_num_ = nthreads;  sched_thread_num_ = nthreads - 1;   // aicpu_executor.cpp:317-318
if (thread_idx >= sched_thread_num_) run_orchestration(...);       // :986-987
if (thread_idx <  sched_thread_num_) resolve_and_dispatch(...);    // 闸门 :996，调用 :1013
```

核归属：cluster（1 AIC + 2 AIV）**轮转分配** `cluster ci → thread ci % active_sched_threads_`（`scheduler_cold_path.cpp:1056-1113`），每线程一个 `CoreTracker`。
**没有专职 resolution 线程** —— 每个调度线程解析自己那些核的完成。

**主循环** —— `resolve_and_dispatch`，`scheduler_dispatch.cpp:891-1555`：

```text
while (true) {
    if (completed_) break;
    if (无运行中的核) handle_orchestrator_exit(...)                    // :1032
    阶段1  check_running_cores_for_completion(...)  轮询自己的核 FIN/ACK // :1046
    阶段1b async_wait_list.poll_and_complete(...)   SDMA/RDMA 延迟完成   // :1105
    阶段2  if (sync_start_pending) { handle_drain_mode(); continue; }   // :1164
    阶段3  dummy_ready_queue.pop_batch(...)         纯依赖任务退休        // :1200
    阶段4  dispatch_ready_tasks(...)                就绪派发              // :1305
    阶段4b try_early_dispatch(...)                  投机预派发            // :1346
    空闲尾 延迟释放、ring 回收、停滞/超时看门狗、SPIN_WAIT_HINT()         // :1383
}
```

**`run_staging_order` 是策略骨架**（`:503-561`），顺序固定为：

```text
MIX-IDLE  ▶  AIC/AIV-IDLE  ▶  MIX-PENDING  ▶  AIC/AIV-PENDING
```

AIC/AIV 的先后由 `thread_idx & 1` 翻转（`:511-515`），避免多线程抢同一形状。

**任务如何变 ready** —— **原子 fanin refcount + 一次性生命周期标志**。

```cpp
// scheduler.h:1160-1170
fanin_refcount.fetch_add(1, acq_rel) + 1 == fanin_count  →  route_ready_once
```

`route_ready_once`（`scheduler.h:1093-1116`）先做 `READY_CLAIMED` 的 CAS，再路由。
`push_ready_routed`（`scheduler.h:544-554`）是路由函数：DUMMY 或谓词失败 → `dummy_ready_queue`；`requires_sync_start()` → `ready_sync_queues[shape]`；否则 → `ready_queues[shape]`。

**还有第二条并行的就绪轨道** —— `dispatch_fanin`，专供早期派发：
`propagate_dispatch_fanin`（`scheduler.h:971-1017`）给每个消费者的 `dispatch_fanin` 加一，达到 `early_dispatch_target()` 就成为预派发候选。

**就绪队列** —— 全部是**无锁有界 MPMC Vyukov 队列**（`ChipReadyQueue`，`scheduler.h:76-391`）：

| 数组 | 位置 | 分区依据 |
| --- | --- | --- |
| `ready_queues[NUM_RESOURCE_SHAPES]` | `scheduler.h:515` | AIC/AIV/MIX |
| `ready_sync_queues[...]` | `:522` | shape，**Tier-0** sync_start 同批 |
| `dummy_ready_queue` | `:526` | 单个，纯依赖任务 |
| `early_dispatch_queues[...]` | `:759` | shape，投机候选 |
| `early_sync_start_queue` | `:773` | 单个 |

**全局共享，不是每线程一份** —— 任何调度线程都能弹出任何 shape 的队列。每线程私有的只有 `CoreTracker` 位图。

**关键判断点**（`dispatch_shape`，`scheduler_dispatch.cpp:309-500`）：

| 位置 | 条件 | 后果 |
| --- | --- | --- |
| `:316` | `entered_drain` | sync_start 排空期间**压制本趟所有派发** |
| `:396-409` | MIX 无匹配 cluster | 推回队列并 `continue` |
| `:417-420` | sync_start + `is_pending` | **sync_start 绝不使用 pending 槽** |
| `:421-432` | `available < logical_block_num` | 进入**全局停机排空**（`enter_drain_mode`） |
| `:435-442` | 核用尽 | 自旋推回剩余批次（**不能丢**），`break` |
| `:451-454` | `claim_block_range(...)` | SPMD 块范围的原子 CAS 认领；返回 0 → `continue` |
| `:462-464` | 未认领完 | **先推回再做昂贵的准备/发布**，让同伴能并发认领 |

`run_staging_order` 的三个门控：

- `:523` **MIX 严格优先** —— 还有 MIX 残留就整趟跳过 AIC/AIV
- `:531` `if (pmu_active) return;` —— **PMU 模式下不用 PENDING 阶段**（单发射要求）
- `:542` `if (!has_idle_in_other_threads(...))` —— 同伴有空闲核时，让它以更低延迟拿走

**MIX 核放置** —— `classify_mix_cluster`（`scheduler_types.h:356-376`）返回三态：
任一被用核的 pending 槽已占用或掩码为空 → **REJECT**；全部空闲 → **RUNNING**；否则 → **PENDING**。

**早期派发的前置条件**（`try_early_dispatch`，`:803-885`）：

```cpp
if (pmu_active || !tracker.has_any_free_slot()) return 0;                  // :816
for each shape: if (ready_sync_queues[s].size() || ready_queues[s].size())
    return 0;   // ← 只有所有正常就绪队列都空时，才做投机派发          // :817-819
```

**sync_start 排空协议** —— `enter_drain_mode`（`scheduler_completion.cpp:503-522`）CAS `sync_start_pending 0 → -1`；
`handle_drain_mode`（`:679-823`）让每个线程通过 **O(log N) 二叉树令牌屏障**确认，由 thread 0 做协调者检查全局可用核数，够了才 `drain_stage_go = 1`，所有线程并行 stage。

**完成观测** —— **轮询每个核的 MMIO COND 寄存器**：

```cpp
reg_load_acquire(core.cond_ptr);  rmb();     // scheduler_completion.cpp:338-343
```

`decide_slot_transition`（`:40-83`）是个**纯函数**，返回 `SlotTransition{running_done, pending_done, ...}`，覆盖 4 类顶层情形（Case 3 再分三个子情形，含"running FIN 且 pending 被早期派发门控 → 完成并提升"这个避免死锁的特例）。

完成链：`complete_slot_task` → `on_subtask_complete`（原子计数）→ 最后一个子任务时 `on_task_complete` 走 fanout 链表 → 延迟释放缓冲（`DEFERRED_RELEASE_CAP = 256`）→ 空闲时排空。

**扩展点** —— **没有函数指针表、没有虚函数**。唯一真正的多态是：

> **`run_staging_order<StageFn, ResidualMixFn>`** —— `scheduler_context.h:407-408`，实现在 `scheduler_dispatch.cpp:503`。
> 这是这层**唯一的策略模板**；正常派发和早期派发传入不同的 lambda。
> **想改 shape/phase 的优先级顺序，从这里下手**（配套还要动 `dispatch_shape` 等，见 §5 方案 2）。

---

### (c) L2 HBG AICPU 调度器

整个编排在**主机**完成，设备侧只负责调度，且**全图常驻**（无槽位/堆回收）。

**执行位置 —— 3S+1P**：`(aicpu_thread_num - 1)` 个持核的 **S 线程**，加**最后一个索引上的专职 resolution（P）线程**。

```cpp
p_thread_idx_ = aicpu_thread_num_ - 1;              // scheduler_cold_path.cpp:773
(thread_idx == p_thread_idx()) ? run_resolution_thread(...) : resolve_and_dispatch(...);
                                                    // aicpu_executor.cpp:371-373
```

S→P 的交接是**每个 S 线程一个 SPSC 环**（`struct CompletedTaskQueue`，`scheduler_context.h:52-93`；数组成员 `sp_queues_[]` 在 `:251`）。

**两个循环**：

```text
S 线程（scheduler_dispatch.cpp:1218+）
  轮询自己的核 → 推入 sp_queues_          ← 不做解析
  排空模式检查
  有界图物化（每轮一个切片）
  dispatch_ready_tasks / try_early_dispatch
  注意 :1381-1385：async 轮询与 dummy 退休都不在 S 上，已移到 P

P 线程（:952-1212）
  遍历所有 sp_queues_ → complete_task()    ← 发布 + 排空唤醒链表
  async_wait_list.poll_and_complete(...)
  dummy_ready_queue 排空
  空闲时：前进看门狗（仅当没有线程持有 RUNNING 任务才锁存）
```

**任务如何变 ready** —— **不是 refcount，是轮询完成 + 侵入式唤醒链表**（`scheduler.h:612-702`）：

就绪判据是"inline fanin 里每个生产者的 `task_states` 字节都是 COMPLETED"。

- `classify_fanin_state`（`:633-645`）**从 `wake_scan_cursor` 反向扫描**（最晚提交的生产者优先 —— 这样能最小化唤醒链表的转移，理由写在 `:617-632`），返回 `-1`（全满足）或反扫中遇到的第一个未满足下标。
- `register_wake`（`:651-670`）把消费者 CAS 推到生产者的 `wake_list_head`；如果生产者已封闭链表（`WAKE_LIST_SENTINEL`），就重新分类并改挂。
- `on_mixed_task_complete`（`:676-702`）：写 `task_states`，封印 ED 发布链表，然后 `wake_list_head.exchange(SENTINEL)` 遍历等待者 —— `fanin_count == 1` 直接就绪，否则重新分类。

**队列容量在主机 bind 时算好** —— `ReadyQueuePopulations::add_task` / `derive_capacities`（`src/common/host_build_graph/host/ready_queue_sizing.cpp:38-87`），取 ≥ 该 shape 全图任务数的下一个 2 的幂（是上界，不是实测峰值）。
稳态完成路径上 **P 是就绪队列的唯一生产者**；启动时的 `classify_partition` 和运行中 S 线程的图物化也会入队，所以队列本身仍是 MPMC。

**HBG 特有的判断点**：

| 位置 | 内容 |
| --- | --- |
| `scheduler.h:579-586` | GRAPH shell 的释放**不能**排在排空闸门后面，所以 `push_ready_routed` 在完成路径上就地处理 |
| `scheduler_dispatch.cpp:1472` | `prepare_graph_task(*prepare_slot, GRAPH_MATERIALIZE_SLICE_TASKS /*=4*/, ...)` —— **每轮只物化一个有界切片**，防止大定义独占线程 |
| `scheduler.h:541-552` | **就绪队列溢出是致命错误**，不是背压 —— `SIMPLER_ERROR_READY_QUEUE_OVERFLOW` |
| `scheduler.h:570` | ED 候选门控读 `ed_flags`，而 **`ed_flags` 主要由主机侧算好**（设备侧只有图物化会为 body root 补算）—— 不同于 TRB 在运行时发现 |

**完成观测** —— 同样是 MMIO COND 轮询，但**解析被卸载**：
S 线程的 `complete_slot_task` 结尾是 `sp_queues_[thread_idx].push(&slot_state)`（`scheduler_completion.cpp:196`），**不解析、不计数**。P 弹出后才做 `complete_task` → 唤醒链表排空 → `push_ready_routed`。

**没有回收步骤** —— 全图常驻，没有 `last_task_alive`、没有 ring 推进、没有 CONSUMED 状态。

---

### (d) a5 HBG 常驻 AIV 调度器

**执行位置 —— 在 AICore 上。** 每个 AICore worker 跑同一个常驻程序，**每个硬件 cluster 选举一个 AIV 当 Scheduler**，另外两条 lane（AIC + 另一个 AIV）是纯 Executor。

```cpp
// aicpu/aicore_lifecycle.cpp:248-268
scheduler_worker = min(physical_core_id of {aiv0, aiv1});   // 物理核号小的那个
scheduler_index  = cluster;
scheduler_count  = aic_count;                                // == cluster 数
```

每个 Scheduler 管 3 条 lane × 2 个派发槽（`SCHEDULER_PENDING_SLOT_COUNT == 2`）。
**AICPU 只是监工**：初始化屏障、发布 context、轮询 `resolved_task_count`、关机屏障。

**主循环** —— `run_ready_dispatch_loop`，`aicore_executor.cpp:291-510`：

```text
while (true) {
    if (read_reg(DATA_MAIN_BASE) == AICORE_EXIT_SIGNAL) break;
    if (++poll % 64 == 0 && scheduler_error) return false;
    // ── Scheduler 半边（仅 Scheduler lane 执行）
    if (scheduler_worker) {
        scheduler_refresh_ready_inbox(...)                      // 收件箱
        scheduler_drain_deferred_aiv_to_peer(...)               // 把自己的活让给同伴
        scheduler_service_cluster_completions(...)              // 完成 + 直接回填
        scheduler_fill_cluster_normal_slots(...)                // 派发
        ... 再排一次 deferred ...
    }
    // ── Executor 半边（每个核都跑，含 Scheduler 自己）
    取到任务 → execute_task(payload) → 发布完成 → continue
    if (scheduler_progress) continue;
    scheduler_flush_completions(...);                           // 批量提交计数
    local_backoff(...);   backoff <<= 1  (8 → 32)
}
```

**bootstrap 本身就是一趟调度**（`bootstrap_ready_graph`，`:141-262`）：每个 Scheduler 分类自己那 1/N 切片 → 路由或挂唤醒 → 发布按核类型分批 → **计数屏障** → 发布就绪目录 → 初始化槽位 → **在 DMB 启动闸门打开前预填第一波**。

**任务如何变 ready** —— **轮询状态 + GM 中的无锁侵入式唤醒链表**，没有 refcount。

```cpp
// scheduler_types.h:775-792
struct SchedulerTaskControl {
    volatile int64_t state;           // BLOCKED=0, DONE=2
    volatile int64_t wake_list_head;  // OPEN=-1, CLOSED=-2
    // 另一条 cache line：next_waiter, next_fanin_index, waiting_producer
};
```

`scheduler_route_task`（`scheduler_ready.h:576-643`）从 `next_fanin_index` 往后走：生产者 `state == DONE` 就跳过；否则**CAS 推到生产者的 `wake_list_head`**；链表已 CLOSED 就前进到下一个 fanin。全部满足 → `READY_TO_ENQUEUE`。

**直接解析捷径**（`:1347-1357`）：完成时第一个新就绪、且核类型与刚释放的槽匹配的等待者，直接作为 `direct_ready` 返回，**绕过收件箱就地回填槽位**。

**就绪队列** —— GM 里**按 scheduler、按核类型的侵入式单链表 + 位图目录**：

- `SchedulerReadyInbox[core_type][scheduler_index]` —— **每 cluster 2 个**（AIC/AIV）
- `SchedulerReadyDirectory` —— 分片位图，**每 64 字节分片管 7 个 owner**（`SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD`）

**弹出/窃取策略**（`scheduler_claim_ready_for_slot`，`:1066-1107`）：

```text
1. 刷新收件箱（把停放的批次接进来）
2. 先弹自己的收件箱              → source = LOCAL
3. 否则读本 scheduler 所在的目录分片，
   在分片内做两轮轮转窃取         → source = STOLEN
4. 推进 victim 游标
```

**窃取是分片局部的（同分片至多 7 个 owner，除去自己 ≤6 个同伴），不是全局的。**
收件箱弹出用 CAS，**最多尝试 64 次**就放弃本轮（`contention_giveup_count`），留到后续迭代重试。

**关键判断点**：

`scheduler_fill_cluster_normal_slots`（`scheduler_dispatch.h:71-244`）—— AIC 和 AIV 两半策略不同：

| 位置 | 内容 |
| --- | --- |
| `:84-87` | **容量与有活双重确认**后才去查询：`has_usable_slot && ready_directory_nonempty` |
| `:182-194` | **AIV 选择策略**：`scheduler_normal_aiv_worker_precedes` —— **非 Scheduler 的 AIV lane 严格优先于 Scheduler 自己那条**；平手则**占用槽少者优先**（最小负载） |
| `:198-199` | Scheduler 自己那条 AIV 的容量被 deferred 队列长度封顶 |
| `:222-229` | **选中的是 Scheduler 自己 → 活被"延迟"进 `SchedulerDeferredAivQueue`**，而不是直接派发；之后再交给同伴或本地执行 |

`scheduler_fill_dispatch_slot`（`scheduler_ready.h:1130-1270`）的**形状准入**（`:1158-1168`）：

```text
非单子任务 or 目标不是 AIC/AIV or 元数据与目标核类型不符 or 不可执行 or 是 gang(MIX/SPMD)
    → UNSUPPORTED_SHAPE → 致命错误
```

**谓词在派发时求值**（`:1203-1213`）：`FAIL` → `function_bin_addr = 0`，Executor 随后空转退休。

**全局形状闸门（主机侧）** —— 判据在 `runtime/scheduler/scheduler_graph.h:58-61`，
调用点 `host/runtime_maker.cpp:1136-1140`，回退在 `:1175-1182`：

```cpp
// scheduler_resident_v0_task_shape_supported
return active_subtasks == 1 && logical_block_num == 1 && !sync_start;
```

**图里只要有任何 MIX / SPMD / sync-start 任务，整个 run 就回退到 legacy AICPU 调度器**。Graph-execution 的 run 也回退。

**完成观测** —— **没有寄存器轮询、没有 mailbox，用 SSBUF 里的 generation 字**：
Executor 发布 → 远端 lane 写 `lanes[lane].completion.publication`；Scheduler 每条远端 lane 一次 SSBUF load 即可发现。
聚合计数 `scheduler_flush_completions` **只在空闲或收尾时**做一次 `fetch_add` —— 把唯一全局竞争的原子操作批量化了。

**扩展点** —— 整个调度器是 `scheduler_ready.h`(1386 行) / `scheduler_dispatch.h`(386) / `scheduler_completion.h`(283) 里的**头文件内联自由函数**。

- `aicore_execute` 是 `__attribute__((weak))`（`aicore_executor.cpp:646`）—— **链接期可覆盖**
- 真正的策略开关是**运行模式**：`SCHEDULER_RUNTIME_MODE_{RESIDENT_PENDING, RESIDENT_READY, LEGACY_GRAPH, LEGACY_UNSUPPORTED_SHAPE}`

> **⚠️ 已铺设但未启用的 gang 机制** ——
> `SchedulerGangCoordinator` / `Cohort` / `Participant` / `Command`（`scheduler_types.h:809-880`）
> 已经排好布局、预留空间、零初始化，主机侧也发布了、绑进了 `SchedulerLocalConfig::gang_coordinator_offset`，
> 但**唯一的活代码只有一行优先级位 OR**（`scheduler_ready.h:488-492`）。
> **常驻循环里没有任何 gang 派发路径** —— 这显然就是留给 MIX/SPMD/sync-start 的扩展位。

---

## 4. 什么是真共享，什么是拷贝

**真正共享（一份代码，两个架构都编译）：**

`src/common/host_build_graph/` —— `graph_execution.h`、`runtime_types.h`、`types.h`、`task_id.h`、
`dep_compute.h`、`ready_queue_sizing.h`、`runtime_ops.h`、`host/ready_queue_sizing.cpp`、
`host/orchestrator.cpp`(3354 行)、`host/dep_gen_host_graph.cpp`、`device/graph_execution.cpp`。
各 HBG 的 `build_config.py` 用 `SHARED = "../../../common/host_build_graph"` 接进来。

`src/common/hierarchical/` **只被 (a) 使用**，(b)/(c)/(d) 都不 include 它。

**拷贝血缘（独立文件，近乎相同）—— 用 `diff | wc -l` 量过：**

| 对比 | 差异行数 |
| --- | --- |
| a2a3-HBG vs a5-HBG 的 `scheduler_dispatch.cpp` | **2** |
| 同上 `scheduler_completion.cpp` | **11** |
| 同上 `scheduler.h` | **13** |
| a2a3-TMR vs a5-TMR 的 `scheduler_types.h` | **0（完全相同）** |

→ **a5 HBG 的 legacy AICPU 调度器基本就是 a2a3 HBG 那个的克隆。**

**同一架构内 TRB vs HBG** 差异很大（`scheduler.h` 1743 行、`scheduler_dispatch.cpp` 1017 行、`scheduler_completion.cpp` 241 行），因为就绪模型（refcount vs 轮询+唤醒链表）和线程模型（N-S vs 3S+1P）不同 —— 但**派发机械部分是逐行的兄弟**：`CoreTracker`、`dispatch_shape`、`run_staging_order`、`prepare_block_for_dispatch`、`decide_slot_transition`、sync_start 排空协议、早期派发门铃协议。

**真正不同的算法有 5 处**：

1. **就绪判定**：计数器+CAS (a, b) vs 轮询单调状态+唤醒链表 (c, d)
2. **完成传输**：主机 mailbox 轮询 (a) / MMIO COND 寄存器 (b, c) / SSBUF generation 字 (d)
3. **解析位置**：同线程 (a, b) / 专职 P 线程 (c) / 派发的 Scheduler 核自己 (d)
4. **放置**：固定目标 id (a) / 位图 CoreTracker + shape 优先级 (b, c) / 最小负载 + 分片内窃取 (d)
5. **回收**：ring/槽位回收 (a, b) vs 无 (c, d)

---

## 5. 新增一个调度器该怎么做

### 方案 1 —— 新建一个 runtime（影响面最大，边界最干净）

建 `src/<arch>/runtime/<new_name>/` 并放一个 `build_config.py`。

**自动发现，不需要注册**：`simpler_setup/platform_info.py:57-62` 会自动发现任何
`src/<arch>/runtime/<name>/build_config.py`；`runtime_builder.py` 编它的 host/aicpu/aicore 目标；
`kernel_compiler.py` 读它的 orchestration 段。**树里没有别的地方需要知道它的存在。**

要从最近的兄弟镜像过来的文件：`runtime/scheduler/`、`aicpu/aicpu_executor.cpp`、
`aicore/aicore_executor.cpp`、`host/runtime_maker.cpp`、`orchestration/orchestration_api.h`，
外加 `runtime/runtime.h` / `runtime/runtime_types.h`（这两个只有 TRB 有；HBG 从
`src/common/host_build_graph/` 取同名头文件）。
再加 `tests/ut/cpp/<arch>/runtime/<new_name>/CMakeLists.txt` 和 `tests/st/<arch>/<new_name>/` 下的场景测试。

### 方案 2 —— 在现有 L2 调度器里换策略（改动最小，杠杆最高）

**唯一的咽喉是 `SchedulerContext::run_staging_order<StageFn, ResidualMixFn>`**：

```text
a2a3 TRB:  runtime/scheduler/scheduler_context.h:407-408  +  scheduler_dispatch.cpp:503-561
a2a3 HBG:  runtime/scheduler/scheduler_context.h:460       +  scheduler_dispatch.cpp:533-591
a5   HBG:  runtime/scheduler/scheduler_context.h:452
```

它已经把**数据源**（哪个队列数组）和**残留判据**作为可调用对象参数化了。
加第三个模板参数来传 shape/phase 顺序，或者再写一个同形状的函数，就能让新策略与现有策略共存。

配套要动的：`dispatch_shape`、`CoreTracker::get_dispatchable_cores`、`has_idle_in_other_threads`；
如果新增队列，还要改 `push_ready_routed` 和 `ReadyQueuePopulations::add_task`（容量模型）。

### 方案 3 —— 补上 a5 常驻调度器的 gang 路径（明摆着留的洞）

见上文 §3(d) 的警告框。要动的文件：

```text
scheduler_dispatch.h      加一个 scheduler_fill_cluster_gang_slots
scheduler_completion.h    cohort 退休
scheduler_ready.h         gang 就绪发布/认领
aicore_executor.cpp:335-366   循环里加一个阶段
scheduler_graph.h:58-61   放宽形状闸门判据（调用点 runtime_maker.cpp:1136）
tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_scheduler_contracts.cpp
```

做完就能去掉 `scheduler_resident_v0_task_shape_supported` 那个回退。

### 方案 4 —— 新的主机侧（L3）放置策略

`Scheduler::Config`（`src/common/hierarchical/scheduler.h:87-122`）本来就是回调集合，
新策略最自然的做法是往里加一个钩子（像已有的 `active_run_cb` / `early_launch_run_cb`），
在 `worker.cpp:189-214` 接线。
如果是新**传输**，继承 `WorkerEndpoint`（`worker_manager.h:401-478`）—— 那是这层唯一要实现的真接口。

### 具体要实现哪些"接口"

| 目标 | 要实现的东西 |
| --- | --- |
| (a) L3 | `WorkerEndpoint`（虚）和/或 `Scheduler::Config` 回调 |
| (b)/(c) AICPU | **没有接口** —— 照着 `SchedulerContext` 的公开面镜像：`pre_handshake_init`、`handshake_partition`、`post_handshake_init`（HBG 是 `classify_partition` + `on_graph_attached`）、`resolve_and_dispatch`、`run_resolution_thread`(HBG)、`shutdown`、`deinit`、`bind_runtime` |
| (d) AICore | **没有接口** —— 镜像这组自由函数：`scheduler_initialize_local_config`、`bootstrap_ready_graph`、`run_ready_dispatch_loop`、`scheduler_flush_completions`，外加一个 `scheduler_plan_layout` 区域 |
| 任何设备调度器的主机契约 | `RuntimeOps`（`src/common/host_build_graph/runtime_ops.h:47-99`）和 `derive_ready_queue_capacities` |

> 注意：`runtime_ops.h` 虽然是个真函数指针表，但它是 **orchestration-so ↔ runtime 的 ABI**
> （submit_task、scope_begin/end、graph_begin/prepare/end/commit…），**不是调度策略表**。

---

## 6. 开发时的几个提醒

- **改 TRB 或 HBG 要同步两个架构** —— 它们是拷贝血缘；`.claude/rules/codestyle.md` 规则 10 就标识符重命名明确要求同一个 commit 落到兄弟树，其余改动同理，否则两棵树会静默漂移。
- **不要在 AICPU 热路径上打日志** —— `codestyle.md` 规则 7：per-task 的 `LOG_*` 会拖慢 AICPU op 到触发 op-execute 超时，**反而掩盖你想观察的行为**。
- **派发路径上不准 sleep** —— 规则 5，任何层级都适用；只能自旋或阻塞在唤醒原语上。AICPU 上连 `yield()` 都不行。
- **提优化前先 grep [`docs/investigations/`](../investigations/README.md)** —— 里面有 28 篇"测过但否决"，其中好几篇正是调度相关（跨任务批量发布、FIN 排序 dsb 收窄、AICore 直接 MMIO、COND vs GM 通知）。
- **每个 runtime 自己的 `docs/RUNTIME_LOGIC.md`**（a2a3 HBG 441 行、a5 HBG 534 行；两个 TRB 分别 937 / 965 行）包含设计意图，动手前值得读。

## 7. 相关测试

| 调度器 | 测试位置 |
| --- | --- |
| (a) L3 | `tests/ut/cpp/common/hierarchical/test_scheduler.cpp` |
| (b) TRB | `tests/ut/cpp/a2a3/runtime/tensormap_and_ringbuffer/test_scheduler_state.cpp`、`test_ready_queue.cpp` |
| (c) HBG | `tests/ut/cpp/common/host_build_graph/`（20 个文件，含 `test_hbg_ready_queue_seed.cpp`、`test_hbg_scheduler_drain.cpp`、`test_hbg_core_tracker.cpp`、`test_hbg_ed_qualification.cpp`） |
| (d) a5 常驻 | `tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_scheduler_{bootstrap,ready,dispatch,contracts}.cpp` + `hbg_scheduler_test_support.h` |

这些单测**不需要硬件、不需要 GCC** —— 调度器源码是可移植 C++，测试时编到主机，只把碰硬件的原语换成桩（`tests/ut/cpp/support/`：寄存器、缓存维护、设备时钟、设备日志）。这是开发新调度器时最快的反馈回路。
