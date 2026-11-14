# Encoder SWA Bounded Replay —— NPU 适配设计（A3 → A5）

- 版本：v1.2（2026-11-19，审查修复与回归）
- 基线：本仓库 `pr-49005-dsv41`
- 对照基线：[01_gpu_bounded_replay_architecture.md](01_gpu_bounded_replay_architecture.md)
- 前置：decoder replay M1 已落地（encoder replay 的作用就是补回 decoder replay
  没有写的 late-layer SWA 窗口）
- 默认：NPU 上默认关闭，显式 `--enable-encoder-swa-bounded-replay` 开启

---

## 1. 目标与原则

- 复用 GPU 的调度/worker/fold 逻辑：
  `fold_encoder_swa_replay`、`run_encoder_swa_replay`、
  `WindowLayout`/`window_layout`（纯 torch fallback）、
  `resets_swa_window`/`replay_len`/`folded_replay_rows`、
  `swa_gates_prefix_match=False` 全部设备无关。
- NPU 差异只在两处：**请求窗口的行格式/内存**与**注意力如何消费窗口布局**。
- 不新增 `is_npu`：encoder 开关接平台谓词；窗口实现通过池的
  `_build_request_window` 接缝注入。
- E1 只做 eager fold；NPU 没有 prefill graph，E2 无内容；PD/DSpark 放 E3。

## 2. 现状差距矩阵

| # | 机制 | GPU 基线 | NPU 现状 | 影响 | 阶段 |
|---|---|---|---|---|---|
| 1 | 开关解析 | `deepseek_v4_hook.py:333-351,374-414`，CUDA/gfx950 | NPU 不支持 | 接平台谓词，默认关 | E1 |
| 2 | 请求窗口内存 | `RequestWindowSpec` + `RequestWindowMemory`，基于 `KVLayout.page_bytes` | NPU `_window_kv_layout()` 返回 `NPUDSV4KVLayout`，无 `page_bytes` API | 必须用行式窗口 | E1 |
| 3 | 窗口拷贝/提交 | Triton `copy_packed_rows`/`commit_packed_rows` | NPU 无对应 kernel | 需要 torch 行式实现 | E1 |
| 4 | `window_layout` | Triton（CUDA）或纯 torch fallback | fallback 可复用 | 无 | E1 |
| 5 | 池访问 | `get_swa_buffer`/`set_swa_buffer` 走 paged SWA 池 | NPU override 绕过窗口 | 必须切到 workspace | E1 |
| 6 | 窗口注意力 | `request_window_layout` 进入 CUDA metadata | NPU 无窗口模式 | A3/A5 索引语义不同 | E1（最大未知） |
| 7 | fold 压缩写槽 | `_compressor_out_loc` 把 replay 行置 0 | NPU 直接从 `out_cache_loc` 推导 | 会覆盖缓存 K | E1 |
| 8 | 池尺寸 | `kv_bytes+16` | A3 BF16 行 1024 B | 会低估；P1-04 已接入精确字节 | E1 |
| 9 | Radix gate | 由 `request_window is not None` 推导 | NPU 子类继承同一规则 | 无 | E1 |
| 10 | PD | CUDA P 角色支持 | NPU 未验证 | 先拒绝 | E3 |
| 11 | DSpark | 已支持 | NPU 未与窗口联调 | 先验证 | E3 |

> 实现状态（2026-11-14）：第 11 行（DSpark）与行式 request window 已落地
> 并真机验证；PD 仍未落地。2026-11-19 的审查修复见 §8.7，详见 §8。

## 3. E1 —— A3 eager fold

### 3.1 开关与兼容矩阵

`arg_groups/deepseek_v4_hook.py`：

- 自动开启：平台谓词
  `current_platform.supports_encoder_swa_bounded_replay(model_type=...)`；
  NPU 默认仍为关闭，需显式开启。
- 兼容性沿用 GPU 列表：prefill graph 必须 `DISABLED`/`BREAKABLE`
  （NPU 默认 `DISABLED`）；无 CP、无 unified memory、无 mixed chunk、
  无 LoRA、无 session radix；投机只允许 DSpark（其他算法拒绝）。
- E1/E3 额外拒绝（NPU）：PD（prefill 和 decode 都先拒绝）、图像/多模态
  请求（沿用共享校验：`tokenizer_manager.py:1377-1393`、
  `schedule_batch.py:2351-2371` 已对非 CUDA 图像拒绝）。DSpark 已在
  E3 打开（draft ring + `swa_valid_from`）。
- chunk 预算沿用共享校验：`chunked_prefill_size >= 128 + page_size`，
  `max_running_requests > 0`（`deepseek_v4_hook.py:333-351`）。

### 3.2 NPU 行式请求窗口

新增 `hardware_backend/npu/dsv4/npu_request_window.py`：

```text
NPURequestWindowSpec(msgspec.Struct):
  num_slots, layers, page_size, capacity, workspace_rows,
  row_dim, row_dtype, kv_layout(NPU enum, 仅标识)
  nbytes = layers*(slots*capacity + page_rows)*row_bytes
        + workspace_rows*row_bytes
        + layers*tag_rows*8

NPURequestWindowMemory:
  state      [layers, num_slots*capacity + page_rows, row_dim]
  tags       [layers, num_slots*capacity + page_size]
  workspace  [workspace_rows, row_dim]

NPURequestWindow:
  activate / activate_idle / reset / initialize_dummy_history
  buffer(layer)   # history_src -> history_loc 的行 gather（torch）
  commit(layer)   # workspace -> req*capacity + pos%capacity 的行 scatter
  row_loc / page_indices

NPURequestWindowController:
  init_prefill / init_decode / init_verify
  init_decode_direct / init_verify_direct     # direct ring（默认）
  init_graph_metadata / refresh_graph / activate
  _ring_table                                 # slot*ring_pages + p%ring_pages
```

- 行格式来自 `NPUDSV4KVLayout.kv_info(...)`：
  A3 `V41_BF16`（默认 448+64=512 个 bf16，1024 B/token），A5
  `V41_SWA_FP8`（544 B/token）；具体以 `kv_info` 为准。
- 全部 torch 实现，不依赖 Triton；`window_layout` 复用共享纯 torch fallback
  （`dsv41_request_window.py:134-212`）；
- 行 gather/scatter 在扁平 `(pages*page_size, dim)` 视图上做，
  `row_dim` 与 NPU PA_ND 池的 last-dim 一致。

共享改动（实际状态）：

- **S2（已落地）**：`mem_cache/deepseek_v4_memory_pool.py` 抽出
  `_build_request_window(...)` 接缝：默认实现 = 现有 `RequestWindowSpec` +
  `RequestWindowMemory`；NPU 子类 override 为行式实现。
- **S1（未改，非必需）**：`dsv41_request_window.py` 仍在模块顶层导入
  `copy_packed_rows`/`commit_packed_rows`；NPU 环境可正常导入，因此没有
  改成延迟导入。
- **S3（已落地）**：`pool_configurator.request_window_bytes` 通过
  `NPUSRTPlatform.request_window_nbytes` 获取 NPU 真实行字节
  （`NPURequestWindowSpec.nbytes` + 按页对齐的 draft ring），并与原有
  估算取 `max`；GPU 路径不变，继续保留 metadata/scratch 余量。
  页数公式与 `DSV41NPUSingleKVPool.create_buffer` 共用
  `npu_pool_pages(rows, page_size)`（`npu_kv_layout.py`），不再因 ceil/floor
  不一致多预算一页。

### 3.3 池访问改造

`hardware_backend/npu/dsv4/dsv4_memory_pool.py`：

- `get_swa_buffer(layer_id)`（`:813-828`）：窗口存在时返回
  `request_window.buffer(swa_local_layer_id)`（workspace 的 PA_ND 视图），
  否则保持 paged SWA 池；
- `set_swa_buffer(layer_id, loc, cache)`（`:879-904`）：窗口存在时写 workspace，
  不写 `swa_kv_pool`；
- `get_state_buf_infos`（`:1687-1700`）：窗口存在时按窗口状态行返回
  （E3 PD 才用）。

### 3.4 注意力窗口模式（E1 核心，已实现）

窗口元数据统一由 `NPURequestWindowController`（`npu_request_window.py`）
构造，backend 只做委托。`folds_encoder_swa_replay = True`。

**prefill / fold（staging）**：`init_prefill` 构造紧凑
`NPUWindowLayout`（history 只保留可重放位置，新行紧随，floor 来自
`encoder_swa_row_floor`），设 `fm.request_window_layout` /
`fm.swa_loc = layout.write_loc` / `fm.dsv41_window_kv_lens =
layout.kv_lens`。A3 窗口侧用隐式 causal 窗口，`seqused_ori_kv` 传
**staged span**（不是完整 `seq_lens`），compressed 侧仍用完整长度；
显式索引（`ori_mask_mode=0`）在 `has_cmp_kv=True` 时会被 A3 算子拒绝
（`ori_topk` 必须为 0），因此不作为 ratio 1/2 层方案。A5 仍读
`dsv41_win_sparse_indices/topk_length`。

**decode / verify（direct ring）**：`init_decode_direct` /
`init_verify_direct` 设 `window.set_phase(prefill=False)`，用
`make_ring_page_table`/`RingPageTableCache`（`slot*ring_pages + p%ring_pages`，宽度
`ceil(seq_lens/page)+1`，多一页覆盖 speculative step 的 `seqused` 扫描）
作为 `fm.swa_page_table`，`fm.swa_loc` 为环行；不 staging、不 commit。
verify 的 dead 行 `swa_loc=-1`，由 `set_swa_buffer` 映射到未使用的 sink
行（固定 shape，无 host sync）。K 直接写入请求环，attention 直接读环。
prefill 仍保留 staging（fold 的窗口 floor 需要它）。

**公共点**：`get_swa_out_cache_loc` 窗口存在时优先返回
`layout.write_loc`（staging）或缓存的 `fm.swa_loc`（direct）；ratio
元数据在 `encoder_swa_compress_rows` 存在时把 replay 行压缩写槽置 0；
`_dsv41_index_k_store` 的 None guard 兼容 `encoder_swa_replay`。

**布局激活时机**：`init_forward_metadata` 构建 full metadata、再构建 tail
metadata 之后，会重新激活 **full** 布局；`enter_late_layer_tail` /
`exit_late_layer_tail` 分别激活 tail / full 布局。mQA 在 attention forward
之前通过 `store_cache()` 预写 SWA，因此必须在预写前就绑定当前层要用的布局；
仅靠 `forward()` 内的激活不够。dead 行在 prefill 阶段写 workspace 的最后一行
scratch（workspace 至少留一个 window 的余量），在 direct ring 阶段写
`window.sink_row`；payload 在设备端清零（无 host sync）。

### 3.5 窗口生命周期

**prefill / fold（staging）**：

```text
# metadata init 构建 full 布局；构建 tail metadata 后重新激活 full
window.activate(full_layout)
# enter_late_layer_tail 切换为 tail 布局；exit 时切回 full
window.activate(tail_layout)
window.buffer(layer)             # attention 前：历史行 -> workspace
  -> ori_kv = workspace view
  -> 窗口索引 = layout.indices / layout.lengths
  -> 新行写 workspace（layout.write_loc）
window.commit(layer)             # attention 后：workspace -> 请求环
```

**decode / verify（direct ring）**：

```text
window.set_phase(prefill=False)
fm.swa_loc = row_loc(req, positions)     # 直写请求环
store_cache -> set_swa_buffer(环)         # dead 行写 sink
attention(ori_kv=ring, seqused=seq_lens, ring page table)
# 无 commit
```

`initialize_dummy_history()` 仍由 `base_runner.py` 在图捕获前调用；NPU
无 prefill graph，draft worker 无窗口，controller 对 window=None 不实例化。

### 3.6 Radix / 调度 / worker

- `swa_gates_prefix_match=False` 由
  `kv_cache_builder.py:396-405` 根据 `pool.request_window is not None` 自动
  生效（NPU 池是 `DeepSeekV4TokenToKVPool` 子类）；
- `resets_swa_window` / `replay_len` / `folded_replay_rows` 已在共享代码中
  按 `folds_encoder_swa_replay`/feature flag 调用，无需改动；
- `tp_worker.py:642-699` 的 fold / drop rows 逻辑复用。

### 3.7 E1 验收

- 单层 parity：fold 后窗口注意力结果 == 全量重算同窗口（A3 恒等页表假设的
  go/no-go）；
- 端到端：相同前缀重复请求（命中）、多轮对话前缀增长、混合不同请求，
  贪心输出与关闭 encoder replay 的 full recompute 一致；
- ratio-2 层（V4.1 第 2–19 层）的压缩状态在 fold 下重跑是幂等的；
- `cached_tokens > 0` 且 `swa_gates_prefix_match == False`；
- 与 decoder replay 同时开启时的组合测试。

## 4. E2 —— 图回放（暂无线程）

NPU 目前没有 prefill graph（`cuda_graph_config.py:119-123`），encoder replay
的 prefill-graph fold 路径（`prefill_cuda_graph_runner.py:2179-2183`）不可达。
E2 不实现，只在文档中保留"若未来 NPU 支持 prefill graph，需要把窗口布局/
workspace 地址放进固定缓冲并原地刷新"的约束。

## 5. E3 —— PD / DSpark

- **PD（已实现并验证，2026-11-19）**：P 的 SWA 状态由
  `StaticKVStorage.resume_units` 读取；窗口存在时走
  `NPURequestWindow.page_indices(slot, begin, end)`，按位置升序把窗口环行
  作为 `StateType.SWA` 的 unit ids 发出。D 侧同样调用 `resume_units`，
  但窗口不存在，返回的是 paged SWA 页号，因此两端按位置一一对应。
  池的 `get_state_buf_infos()` 在 encoder replay 下把
  `request_window.state` 当作 SWA 池注册，item 字节按
  `swa_page_size // window.page_size` 换算，与 D 的 paged SWA 对齐。
  `platforms/npu.py` 的 PD 拒绝已移除；A3 上 P/D + DSpark + encoder replay
  冷启动与重复前缀命中均通过（见 04 §8.9）。
- **DSpark（已实现并验证）**：draft pool 走 per-request SWA ring，
  `swa_valid_from` 限制窗口下界，A3 用 ring 页表把逻辑位置映射到环行，
  target verify 的窗口由 `build_verify_layout` staging 或 direct ring
  消费；resolver 允许 encoder+DSpark。

## 6. 共享代码改动清单（E1）

| # | 文件 | 改动 | 理由 |
|---|---|---|---|
| S1 | `mem_cache/dsv41_request_window.py:7-10` | Triton 导入延迟化 | NPU 可复用 `WindowLayout` |
| S2 | `mem_cache/deepseek_v4_memory_pool.py:1229-1258` | 抽出 `_build_request_window` 接缝 | NPU 行式窗口注入点 |
| S3 | `model_executor/pool_configurator.py:1294-1359` | 窗口字节由池报告 | A3 BF16 行 1024 B，不能硬编码 |
| S4 | `model_executor/runner/base_runner.py:744-745` | 视验证结果加 None 保护 | draft worker 可能无窗口 |
| S5 | `arg_groups/deepseek_v4_hook.py:333-351,374-414` | 平台谓词 + NPU 拒绝项 | 开关解析 |

状态：S2/S3/S4/S5 已落地；S1 未改（NPU 环境可导入，非必需）。
NPU 专属校验已进一步移到 `platforms/npu.py` 的
`validate_encoder_swa_bounded_replay_args` 与
`decoder_replay_dp_moe_backends`，共享 resolver 不再新增 `is_npu` 分支。

## 7. 风险与开放问题

1. **A3 窗口索引语义（已解决）**：A3 `SparseFlashMla` 用 `seqused_ori_kv`
   推导位置；prefill staging 传 staged span，decode/verify 用 direct ring
   页表 + 完整 `seq_lens`。显式索引模式在 `has_cmp_kv=True` 时被拒绝，
   因此不作为 ratio 1/2 方案。单层 parity 与真机结果见 §8.3/§8.4。
2. **ratio-2 压缩状态幂等性（待验证）**：fold 把 replay 行重新纳入 extend，
   压缩器会用 `start_pos = p-r`、`seqused = n+r` 重跑；需单独验证与原始
   写入逐位一致。
3. **窗口内存布局（已解决）**：行式 gather/scatter 按 NPU PA_ND 行格式实现，
   不复用 packed-page kernel。
4. **PD/DSpark**：DSpark 已实现并验证；PD 的窗口状态跨机传输仍未实现，
   resolver 继续拒绝。
5. **DP attention**：M1/E1/E3 的 A3 验证均在 `attn_dp_size == tp_size`
   （attention TP 1）+ DeepEP 下完成；其他 DP/MoE 组合仍需逐项验证。

---

## 8. 实现与验证状态（2026-11-19，A3）

开发目录：`/home/w00889861/dsv4/sglang_bouned_replay/sglang`
（分支 `pr-49005-dsv41`）；验证脚本在 `/home/w00889861/dsv4/scripts/dsv41/`。

### 8.1 已落地

- **NPU 行式请求窗口 + 控制器**
  `hardware_backend/npu/dsv4/npu_request_window.py`：
  `NPURequestWindowSpec/Memory/View` 与 `NPURequestWindowController`。
  prefill 用紧凑布局（`NPUWindowLayout`/`build_prefill_layout`）；
  decode/verify 默认 **direct ring**（`make_ring_page_table`/`RingPageTableCache` +
  `seqused=seq_lens`），旧 staging 布局（`NPUDecodeWindowLayout`/
  `build_verify_layout`）保留为 `_DIRECT_RING_DECODE=False` 的回退。
- **DSpark ring 辅助** `hardware_backend/npu/dsv4/dspark_ring.py`：
  `draft_swa_ring_size` / `draft_ring_page_table` /
  `seed_draft_graph_page_table`。
- **池接缝**：`DeepSeekV4TokenToKVPool._build_request_window` 默认行为不变，
  NPU 子类 override 为行式窗口；NPU 池 `get_swa_buffer`/`set_swa_buffer`
  在 prefill 读写 workspace；direct 模式下读/写请求环，`-1` 写槽映射到
  未使用的 sink 行；`NPURequestWindowMemory.reset_rows` 补齐共享接口。
- **Allocator**：`kv_cache_configurator` 的 effective hybrid 判定接入
  `needs_paged_swa_allocator`，并统一把 `is_draft_worker` 传给 NPU 池
  （draft 才能正确建成 SWA ring）。
- **后端**：窗口元数据委托给 controller；`get_swa_out_cache_loc`
  窗口/cached 分支；direct 模式 K 直写环、无 commit；staging 模式
  attention 后 commit（eager 走 masked 快路径，graph capture 走固定
  shape + sink）；`_build_dsv41_ratio_metadata` 对
  `encoder_swa_compress_rows` 置 0 槽；`folds_encoder_swa_replay`。
- **Resolver/门控**：NPU 专属校验移到
  `NPUSRTPlatform.validate_encoder_swa_bounded_replay_args`
  （仅 DSpark、拒绝 PD、每 rank 单请求）；DP MoE 后端由各平台
  `decoder_replay_dp_moe_backends()` 提供；`supports_*` 只按模型类型与
  arch（A3）判断，没有独立的 readiness 开关。
- **NPU graph（2026-11-17）**：`ensure_hccp_initialized` 在第一个 custom
  AICPU metadata kernel 前初始化 HCCP（放在 `_init_dsv4_graph_buffers`，
  因为它早于 `NPUCudaGraphBackend.__init__`）；A3 无 replay 与 E3
  target verify graph capture 均通过，resolver 不再要求
  `--disable-cuda-graph`；DP idle replay 的 commit mask 有保护。DSpark
  draft ring 的图捕获已打开（稀疏索引/页表在 replay 前原地刷新），
  真机验证待补。

### 8.2 真机验证（A3，TP16/DP16 + DeepEP + eager decode，DSpark 关）

- GSM8K 50 题（sgl-eval CLI，thinking off）：**1.0000**（同配置 encoder off
  同为 1.0）。
- 共享 system、不同问题、32 条（32K system + 4K question）：命中
  **47.08%**（567,936 / 1,206,263）；encoder off 为 0%。
- 同一 prompt 重复 20 次：fold 路径执行，命中从 128 到 1024 tokens。
- 事实型 prompt 正确（Paris / 4 / 0°C, 32°F）。
- 严格 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` 下无泄漏告警。
- 单测：`test_npu_dsv41_request_window.py` 15 例；`unit_sweep.sh` 13/14
  文件全绿（另 1 个为基线已知失败）。

> 注意：本节的命中测试只校验命中率与精度（GSM8K 均为 prefix=0 的新请求），
> **fold 命中请求的输出正确性当时未做配对校验**。E3 联调发现窗口绝对寻址
> 缺陷并修复，见 §8.4.1。

### 8.3 验证中发现的关键契约（A3）

1. **显式窗口索引必须 `< seqused_kv`**：`SparseFlashMla` 会静默丢弃越界索引
   （窗口全空、输出全 0）。CUDA 风格的固定 128 行 history 偏移不适用于 A3；
   必须用紧凑行布局，并有 CPU 回归测试。
2. **块表按 `max(layout 页数, ceil(seq_lens/page))` 分配**，但超出 layout
   页数的表项必须指向已初始化的 page 0；恒等回绕会读到未初始化的 workspace
   页，使 decode 在 128 位置边界后退化。
3. **decode/verify 用 direct ring**：per-request ring 页表
   `slot*ring_pages + p%ring_pages`，`seqused=seq_lens`；页表宽度需多留一页
   以覆盖 speculative step 造成的 `seqused` 偏移（少了会 vector core
   exception）。旧 staging 路径（升序 gather 到 workspace）保留为回退。
4. **eager/graph commit 分相**：eager 用动态 mask 只 scatter 有效行；
   graph capture 用固定 shape 并把无效行写到 sink row（同 `index_copy_`
   目标重复，最终内容任意但安全）。
5. **A3 metadata 必须 `< seqused_kv`**：显式索引路径被算子静默丢弃越界项；
   direct/staging 的隐式窗口以 `seqused` 为界，紧凑布局与页表余量共同保证
   不越界。

### 8.4 E3 —— DSpark / verify（2026-11-14，A3）

E1 验证（§8.2）只覆盖了 **prefix=0 的新请求**；命中 fold 的输出当时
未做 rank 配对校验。E3 联调时发现并修复了以下契约：

1. **A3 窗口寻址**。A3 `SparseFlashMla` 用 `seqused_ori_kv` 推导每行
   query 的位置。prefill/fold staging 传 **staged span**
   （`NPUWindowLayout.kv_lens`），query 相对 workspace，compressed 侧仍用
   完整 `seq_lens`；decode/verify 的 direct ring 传完整 `seq_lens`，
   由 ring 页表映射到环行（见 item 6）。显式索引（`ori_mask_mode=0`）在
   `has_cmp_kv=True` 时被算子拒绝（`ori_topk` 必须为 0），因此不能作为
   ratio 1/2 层的替代。
2. **NPU 池构造必须传 `is_draft_worker`**。此前 NPU 分支丢弃该参数，
   DSpark draft 池被建成 target，直接拿到 request window（而不是
   draft SWA ring），首次 prefill 注入即 `IdSpaceKind.SLIDING_WINDOW`
   KeyError。
3. **DSpark draft ring**：
   - `swa_valid_from` 限制 draft 窗口下界（命中请求的 prefix 以下行
     不属于该请求）；
   - A3 draft 用 ring 页表把逻辑位置映射到
     `slot * ring_pages + page % ring_pages`；
   - `get_swa_out_cache_loc` / graph refresh 直接给出 ring 行；
   - `set_swa_key_buffer_radix_fused_norm_rope` 跳过 `-1`（未提交
     verify 行 / 窗口外行），不再 clamp 到 row 0 覆盖别的 slot。
4. **Target verify 窗口**：默认为 direct ring（上下文窗口与 verify 块
   直接在请求环上读写，dead 行写 sink）；staging 回退用
   `build_verify_layout` 把上下文（≤W-1 行）+ verify 块（n 行）暂存进
   workspace，attention 后只提交 live 行。两条路径的 dummy/idle 行都不会
   污染真实 slot。
5. **门控**：NPU 允许 encoder replay + DSpark；PD 仍拒绝；graph 已解锁
   （见 §8.5）。decoder replay 与 encoder replay 的组合路径已用共享
   `apply_encoder_row_floor` 补齐 floor，但仍需单独验证。
6. **decode/verify 改为 direct ring**（默认，`_DIRECT_RING_DECODE=True`）：
   `make_ring_page_table`/`RingPageTableCache` + `seqused=seq_lens`，K 直写请求环，无 staging/commit；
   页表多留一页覆盖 speculative step 的 `seqused` 扫描。旧 staging 路径
   保留为回退。A/B 见 §8.4.1。

#### 8.4.1 真机结果（A3，TP16/DP16 + DeepEP + eager，DSpark block 5）

- GSM8K 50（sgl-eval CLI，thinking off）：**1.0000**；DSpark
  `avg_spec_accept_length` 每 rank **3.475–4.591**。
- 命中路径精度：同一 GSM8K 题（1.5K shared prefix）16 个新请求 +
  16 个命中请求；wave2 全部 `cached-token=1536, replay-token=128`，
  **16/16 答案正确**（`e3_hit_accuracy.py`）。
- 16 rank 配对 fold 探针（`e3_probe.py`）：命中输出全部正确；
  fold 是近似（replay 行的窗口在 replay start 处截断），greedy
  近似平局可能翻 token，因此不要求逐位一致。
- 共享前缀 32 条（32K system + 4K question，256 output）：命中
  **47.08%**（567,936 / 1,206,263），同 E1；accept length
  **3.76–4.76**。
- 无 scheduler 异常、无窗口/池泄漏告警。
- 性能（同机同配置 A/B，direct ring vs workspace staging）：
  GSM8K 50 **84.9s vs 96.2s**（1.0000 两者），prefix 32 条
  **218.3s vs 234.0s**、输出 **37.5 vs 35.0 tok/s**、TPOT
  **187.0 vs 210.7 ms**（约 -11%），命中率同为 47.08%。
  direct 模式让 decode/verify 通过 per-request ring 页表直接读写请求环
  （`make_ring_page_table`/`RingPageTableCache` + `seqused=seq_lens`，页表多留一页以覆盖
  speculative step 的 seqused 扫描），只有 prefill/fold 继续 staging。
  代码结构：`NPURequestWindowController` 收敛窗口元数据/图刷新，
  `dsv4/dspark_ring.py` 收敛 draft ring 辅助，backend 只留薄委托。

### 8.5 decode/verify graph（HCCP 初始化修复，A3 已打通）

- **根因**：CANN/driver 在 custom AICPU metadata kernel 之后才启动 HCCP
  时，会破坏 dispatch，导致 `SparseFlashMlaMetadata` 在 capture/sync 时
  报 507018。PR #4（HEX1A0A/sglang）把修复放在
  `NPUCudaGraphBackend.__init__`；本分支的 `init_cuda_graph_state` 会先
  预建 A3 metadata seed，早于该 `__init__`，所以修复必须提前。
- **修复**：`ensure_hccp_initialized` 做一次 TP all-reduce + synchronize，
  并在 `_init_dsv4_graph_buffers` 最前面调用（graph backend 也调用同一
  幂等 helper）；这样第一个 AICPU metadata kernel 之前 HCCP 已就绪。
- **已验证**：A3 无 replay（DSpark 开）的 target verify capture 通过
  （~11.7s，无 507018）；E3 encoder+DSpark 的 target verify graph 也
  capture 通过。resolver 不再要求 `--disable-cuda-graph`。
- **idle replay**：DP idle 没有 verify 行，`refresh_graph` 现在保留捕获的
  dummy layout，只把 `commit_mask` 清零，避免 0 长度 mask 拷贝。
- **仍待**：E3 graph 的完整精度/性能复跑（首次 fold 探针在 idle replay
  修复前崩溃）；DSpark draft ring 的图捕获已打开但真机验证待补。
  decoder M2 首版已实现并在 A3 验证（见 02 §9）。

### 8.6 待办

- PD 的窗口/`swa_valid_from` 状态传输（Ascend connector）；
- 每 rank 多请求 prefill：已解除单请求 gate（`MAX_RUNNING_REQUESTS`
  不再受 `attn_dp_size` 限制）；A3 32 并发（2/rank）命中路径 32/32、
  `e3_probe 32` FOLD OK 32/32。compact 布局本身按 `req` 分组，无需
  另建 per-request 视图；
- A5（M3：MQSMLA/QLI metadata、fp8/fp4 行格式）；
- 全量 M1/M2/E3 验证矩阵与性能数据（04_verification.md §3）；
- E3 graph 的完整精度/性能复跑（idle replay 修复后）；
- DSpark draft ring graph 的真机验证（捕获门控已打开）。

已补齐（2026-11-15）：

- **decoder replay + encoder replay 组合**已在 A3 真机验证（GSM8K 1.0000、
  共享前缀命中 47.08%、fold 命中 16/16、prompt logprob 逐位一致），
  但首次上机暴露并修复了两个契约缺陷：tail 视图必须携带 decoder replay 的
  窗口 floor（`swa_replay_start`），且 fold 的 `encoder_swa_compress_rows`
  必须重映射到 tail 行。详见 04_verification.md §8.6。
- 请求窗口字节已按 NPU 行格式精确核算（`SRTPlatform.request_window_nbytes`，
  A3 BF16 行；GPU/ROCm 返回 None 保持原估算）。

### 8.7 2026-11-19 审查修复（gpt_review2）

E1/E3 相关修复（`npu_request_window.py` / `dsv4_memory_pool.py` /
`ascend_dsv4_backend.py` / `platforms/npu.py`）：

1. **窗口布局激活**：`init_forward_metadata` 在构建 tail metadata 后重新
   激活 full 布局，`enter/exit_late_layer_tail` 激活 tail/full 布局；
   覆盖 mQA 在 attention 前 `store_cache()` 预写 SWA 的路径。
2. **SWA sink 隔离**：paged target 池的 dead 行写保留页 0（不再写
   `pool.size` 所在的可分配页），draft ring 写 ring 外的备用行，request
   window 写自身 sink/workspace scratch；dead payload 清零。
3. **窗口预算**：draft ring 页数与 `create_buffer` 共用
   `npu_pool_pages()`，修正多预算一页。
4. **PD 门控**：恢复 NPU 上 prefill/decode 的 PD 拒绝，待连接器消费
   `page_indices()` 并验证页大小/页序/`swa_valid_from`；E3-PD 仍未实现。
5. **配置校验收敛**：NPU 平台钩子删除重复的 `is_npu` 早退与 DSpark 校验
   （共享 resolver 已覆盖），只保留 NPU 已确认的限制。
6. **解码 tail 的 D2H**：`_tail_compress_rows` 用 fold extend 的 host 行数
   作为 mask 上界，去掉 `int(rows.max())`。

验证：`unit_sweep.sh` `NEW_FAILED_FILES=0`（唯一失败为基线已知
`test_npu_dsv41_low_ratio_backend.py`）；窗口激活在
`test_npu_dsv41_decoder_tail.py` 有断言；PD 门控单测恢复通过。
