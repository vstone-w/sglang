# GPU Bounded Replay 架构基线

- 基线分支：`pr-49005-dsv41`（本仓库）
- 参考实现：GPU `dsv4` / `dsv4` HIP / TRT-LLM 后端（本仓库已合入代码）
- 外部说明：`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx:100-108`

本文只描述**现状代码**，作为 NPU 适配的对照基线。两个特性共享 "bounded replay"
名字，但语义不同，必须分开理解。

---

## 1. Decoder SWA bounded replay

### 1.1 语义

V4.1 每个请求每层只保留一份 512 宽 KV latent；层内两路读取：

- **压缩路**：只在 source 层产生（V4.1-Flash 为第 2/8/14/20 层），
  层 2–19 为 2:1 压缩，层 20 起为 1:1；
- **滑动窗口（SWA）**：每层按 128 个位置重算，不随上下文增长。

因此 prefill 时第 `late_layer_start = max(kv_source_layer_ids)`（V4.1-Flash 为 20）
之后的 20 层只需要每个请求 extend 的最后 128 个 token（tail），
更早的 prompt 行对 late layers 是死数据。

例外（whole-extend）：需要读取每个 prompt 行的请求跑完整 extend：

- prompt logprob（`extend_logprob_start_lens < extend_len`）；
- 非 DSpark 的 `CaptureHiddenMode.FULL`（要返回每个位置的 hidden state）。
  DSpark 虽然也 FULL capture，但只读 tail 行，因此仍走 tail。

实现：`_late_layer_whole_extend` / `late_layer_whole_extend`
（`layers/attention/deepseek_v4_backend.py:1036-1068`）。

### 1.2 行选择与窗口下界

`late_layer_tail_lens` / `late_layer_tail_layout`
（`kernels/ops/attention/dsv4/attn_metadata_kernels.py:447-528`）：

- 默认 `tail_len[b] = min(128, extend_len[b])`；
- whole-extend 请求 `tail_len[b] = extend_len[b]`；
- 返回 `(token_indices, tail_lens_cpu, swa_replay_start)`：
  - `token_indices` 是 extend 内最后 `tail_len` 行的索引，按请求顺序展开；
  - 单请求时用 `contiguous_start = extend_len - tail_len` 的切片视图；
  - **窗口下界** `floor = max(0, seq_len - 128)`：一个短于 128 的 chunk
    可能读到前一个 chunk 写入的行，所以窗口回退一个窗口，而不是 extend 起点；
  - whole-extend 请求的 floor 为 `max(0, seq_len - extend_len - 128)`，
    即窗口不被截断。

### 1.3 元数据契约

`LateLayerTail`（msgspec frozen，`deepseek_v4_backend.py:992-1033`）：

| 字段 | 含义 |
|---|---|
| `token_indices` | tail 行在 extend 内的索引 |
| `positions` | tail 行绝对位置 |
| `extend_seq_lens` / `_cpu` | 每个请求的 tail 长度 |
| `swa_out_cache_loc` | 每 tail 行的 SWA 写槽 |
| `contiguous_start` | 单请求切片优化 |
| `pad_rows` / `cp_metadata` / `local_lens_cpu` / `req_global` / `pos_global` | prefill CP 布局（NPU 暂不支持） |
| `scatter_indices` / `real_rows_count` | prefill CUDA graph 的定长块布局 |

`DSV4Metadata` 增加 `late_layer_tail`、`prefill_graph_tail`、`real_num_tokens`
（`deepseek_v4_backend.py:1139-1148`）。

后端状态：`tail_forward_metadata`、`tail_source_kv_layer_id`、
`prefill_graph_replay_batch`、`_bcg_tail_*`（`deepseek_v4_backend.py:1365-1381`）。

### 1.4 后端虚函数（模型无条件调用）

| 接口 | CUDA 实现 | 作用 |
|---|---|---|
| `late_layer_num_tokens(batch)` | `deepseek_v4_backend.py:1834-1839` | 本 rank late-layer 行数；`forward_batch_info.py:1929-1942` 在 forward 前跨 DP rank 交换 |
| `_build_late_layer_tail_metadata(batch)` | `deepseek_v4_backend.py:1910-2096` | 用 tail 行调用 prefill 元数据构造器 |
| `write_tail_source_kv(layer, x, positions, batch)` | `deepseek_v4_backend.py:2163-2178` | 在完整 extend 上跑 source 层压缩器（不跑 indexer），并记录 `tail_source_kv_layer_id` |
| `enter_late_layer_tail(batch)` | `deepseek_v4_backend.py:2180-2218` | 保存并切到 tail metadata、request window、CP、非 padding 计数 |
| `exit_late_layer_tail(saved, batch)` | `deepseek_v4_backend.py:2220-2233` | 恢复 |
| `get_swa_out_cache_loc(batch)` | `deepseek_v4_backend.py:4273-4303` | tail 存在时返回 tail 的 SWA 写槽 |
| `can_run_prefill_cuda_graph` / `vote_prefill_cuda_graph` | `1816-1861` | prefill 图容量判定与 DP 投票 |

模型侧：`models/deepseek_v4.py:4160-4333` 在 `i == late_layer_start` 处
`write_tail_source_kv` → `enter_late_layer_tail` → 把
`state/input_ids/positions/hash_ids` 切到 tail 行 → 跑 late layers
（可选用 `DecoderReplayTail` 图回放）→ `exit_late_layer_tail`。
lm_head/logits 路径（`deepseek_v4.py:5119-5160`）也用 tail 长度对齐。

### 1.5 DP attention

- 每个 rank 的 late-layer 行数在 forward 前通过 `late_layer_num_tokens` +
  `all_gather_single(tp_group.cpu_group)` 交换，写入
  `forward_batch.global_late_layer_num_tokens_cpu`
  （`model_executor/forward_batch_info.py:1929-1942`）。
- `late_layer_dp_counts` 决定是否重排 DP 计数（`models/deepseek_common/decoder_swa_bounded_replay.py:16-51`）。
- `enter_late_layer_dp` / `exit_late_layer_dp`（同文件 `54-119`）临时改写
  `global_num_tokens*`、`global_dp_buffer_len`、`dp_padding_mode = SUM_LEN`，
  并把 `num_token_non_padded` 钳到真实 tail 行数。
- GPU 约束：DP attention 要求 `attn_dp_size == tp_size`、`attn_cp_size == 1`，
  MoE A2A 为 `none`/`megamoe`（CUDA）或 `none`（HIP）
  （`arg_groups/deepseek_v4_hook.py:467-488`）。

### 1.6 图回放

**Eager prefill 的行桶图**（CUDA）：

- `DecoderReplayTail`（`models/deepseek_v4_replay_tail.py:136-637`）：
  - `maybe_create` 目前硬门 `is_cuda()`、`max_rows > 0`、非 graph pool borrow
    （`:155-173`）；
  - `plan()` 决定本步是否走图（`:188-240`）；
  - `run()` 把 `HcState`/positions/input_ids 写入静态缓冲，按
    `row_step = SWA_WINDOW` 行桶，调用 `RowBucketGraphRunner.run`
    （`:301-396`）；
  - `_layer_stack`（`:585-637`）复刻模型的 late-layer 循环（CUDA 路径）；
  - DP 时每个 rank 回放最大 rank 的桶；idle rank 走 `run_idle`（`:398-421`）。
- `RowBucketGraphRunner`（`model_executor/runner/row_bucket_graph_runner.py:84-343`）：
  - 每个行桶一张 `BreakableCUDAGraph`，首次使用时**从大到小**捕获；
  - 输入拷贝到静态缓冲，图内 KV 写/元数据切换是 eager break；
  - `SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS` 控制最大行数，
    `SGLANG_DSV4_DECODER_TAIL_GRAPH_MIN_PREFILL_TOKENS` 控制 prefill 图内
    跳出 tail 图的最小 token 数（`environ.py:1930-1935`）。

**Prefill CUDA graph 内嵌 tail**（CUDA BCG only）：

- `run_in_prefill_graph`（`deepseek_v4_replay_tail.py:471-508`）在外层 prefill 图
  的一个 eager break 里跑 tail，通过 bridge buffer 把结果写回；
- 后端负责 tail 块容量、持久 index/position/scatter/swa/loc 缓冲
  （`deepseek_v4_backend.py:1871-1908`）；
- NPU 当前没有 prefill graph（`cuda_graph_config.py:119-123`），这条路径不在
  NPU 适配范围内。

### 1.7 Radix cache 契约

decoder replay 下，late layers 只写了尾部 128 行的 SWA KV。因此前缀命中时
不能直接复用最后 128 个 token 的窗口，必须重算：

- `UnifiedRadixCache.swa_reprefill_tail_tokens`（`mem_cache/unified_radix_cache.py:4184-4207`）
  返回 `sliding_window_size`；
- `_req_radix_key` 把匹配长度截到 `len - reprefill_tail`
  （`mem_cache/common.py:225-244`）；
- 该行为由 `params.decoder_swa_bounded_replay` 控制，
  值来自 `get_exec().features.enable_decoder_swa_bounded_replay`
  （`mem_cache/kv_cache_builder.py:459`）。

### 1.8 DSpark 交互

- bounded replay 只作用于 plain EXTEND（`is_extend_without_speculative()`），
  draft/verify 的 `TARGET_VERIFY` 不走 tail。
- prefill 的 DSpark aux hidden states 按 `dspark_layers_to_capture` 捕获，
  捕获层在 tail 前用 `tail.rows(...)` 裁到 tail 行
  （`deepseek_v4.py:4351-4358`）；
- draft KV 注入通过 `logits_output.hidden_states_token_indices` 映射回调度行
  （`deepseek_v4.py:5119-5167`；`speculative/dspark_components/dspark_kv_inject.py:127-150`）。
  draft 层是纯 SWA（`compress_ratio=0`），只读最后窗口，所以只捕获 tail 行是安全的。
- `DecoderReplayTail` 的图 key 包含 `capture_dspark`，aux 输出随图返回。

### 1.9 显存/池

decoder replay 本身**不新增 KV 池**：tail 行是 extend 行的子集，
`LateLayerTail` 与行桶图只持有视图/图缓冲。策略影响是 late layers 只为 tail 行
写 SWA KV，decode 只读最后窗口。

### 1.10 启动解析与兼容矩阵

`arg_groups/deepseek_v4_hook.py:467-515`：

- 显式 `--enable-decoder-swa-bounded-replay`：检查 prefill graph 必须是
  `DISABLED`/`BREAKABLE`；DP attention 必须满足 1.5 的约束。
- `None`（未显式指定）自动开启条件：
  - 平台为 CUDA 或 HIP；
  - PD 角色：CUDA 上 P/D 都可，HIP 未验证；
  - 无 prefill CP / attn CP / DCP；
  - 不返回 hidden states；
  - 不命中上述不兼容项。
- **注意**：显式分支在平台判断之前 `return`，所以今天在 NPU 上显式开启会通过
  resolver，随后在模型初始化/后端访问 `tail_forward_metadata` 时失败。
  NPU 适配必须把平台能力谓词接进这个分支。

---

## 2. Encoder SWA bounded replay

### 2.1 语义

radix cache 命中后，Main KV 与 Indexer K 可以复用；但请求私有的 SWA 环形窗口
不能复用（可能被别的请求覆写，或者 decoder replay 下根本没写过 late-layer 窗口）。
做法：重跑命中前缀的最后 ≤128 个缓存 token，重建该请求的窗口。

### 2.2 请求窗口（request window）

- `RequestWindowSpec`（msgspec frozen，`mem_cache/dsv41_request_window.py:241-326`）：
  - `num_slots`、`layers`、`page_size`、`capacity = sliding_window + draft_tokens`、
    `workspace_rows`、`kv_layout`；
  - 每层状态页数 `num_slots*capacity/page_size + 2`，tag 行
    `num_slots*capacity + page_size`，workspace 页数 `workspace_rows/page_size + 1`；
  - `nbytes` 按 `kv_layout.page_bytes(page_size)` 计算。
- 内存：`RequestWindowMemory`（`mem_cache/kv/memory/request_window.py`）用
  `FixedMemory` 分配 uint8 页张量，构造 `PackedWindowView`。
- `RequestWindow`（`dsv41_request_window.py:350-465`）：
  - `activate(layout)` 绑定当前布局（捕获后不允许增长 workspace）；
  - `buffer(layer)` 把历史行拷进 workspace；
  - `commit(layer)` 把新行写回请求环并更新 tag；
  - `initialize_dummy_history()` 给图捕获的 dummy forward 提供合法历史。
- 池创建：`DeepSeekV4TokenToKVPool.__init__`
  （`mem_cache/deepseek_v4_memory_pool.py:1215-1265`），仅当
  `enable_encoder_swa_bounded_replay` 且非 draft worker 时创建。
- 容量与尺寸：`pool_configurator.py:1283-1365`（request window 字节数、
  workspace、draft ring）；`request_window_workspace_rows`
  （`dsv41_request_window.py:329-347`）。

### 2.3 两种执行方式

**Fold（CUDA `dsv4`，推荐）**：

- `fold_encoder_swa_replay`（`model_executor/encoder_swa_replay.py:21-30`）：
  对每个 `encoder_swa_reset` 的请求，窗口 reset，把最后 `replay_len(prefix) =
  min(prefix, 128)` 个 token 拼到 extend 前面；
- `_fold_batch`（`:203-268`）重写 `input_ids/out_cache_loc/prefix_lens/
  extend_lens/...`，产出 `FoldedExtend`（keep_rows/scheduled_row/row_floor）；
- `apply_folded_extend`（`:55-57`）把 `encoder_swa_row_floor` 和
  `encoder_swa_compress_rows` 挂到 ForwardBatch；
- `drop_folded_rows`（`:60-88`）把 logits/hidden states 裁回原始调度行；
- 后端必须声明 `folds_encoder_swa_replay = True`。

**单独 replay forward（非 fold）**：

- `run_encoder_swa_replay`（`:101-123`）构造只含 replay 行的 ForwardBatch，
  `fb.encoder_swa_replay = True`，跑一次 forward；
- 后端在 `forward_low_ratio_sources` 里跳过压缩器
  （`deepseek_v4_backend.py:3554`）。

### 2.4 后端契约（fold 路径）

- `encoder_swa_row_floor`：每行的窗口下界（命中请求的 replay 起点），
  用于 SWA 索引构造和 tail 元数据（`deepseek_v4_backend.py:1930-1935,
  3183-3184, 5035-5039`）。
- `encoder_swa_compress_rows`：fold 中属于 replay 的行；压缩器/索引 K 的写槽
  用 `_compressor_out_loc` 把这些行置 0
  （`deepseek_v4_backend.py:3160-3172`），避免覆盖已缓存的压缩 KV / index K。
- `swa_replay_start` 传入 `make_core_attn_metadata`，窗口索引按行下界裁剪。
- 请求窗口布局 `WindowLayout` 由 `window_layout()` 构造
  （`dsv41_request_window.py:93-212`），CUDA 走 Triton `build_window_layout`，
  其他平台有纯 torch fallback。

### 2.5 Scheduler / Worker / Cache

- `resets_swa_window(req)`（`encoder_swa_replay.py:33-36`）：新槽位或 retraction；
- `replay_len(prefix)`（`:39-42`）：`min(prefix, 128)`；
- `folded_replay_rows(batch)`（`:45-52`）计入 prefill 预算
  （`schedule_policy.py:950-955,1637-1641`；`dp_attn.py:616-622`）；
- `TpModelWorker.forward_batch_generation`（`managers/tp_worker.py:642-699`）
  按 `folds_encoder_swa_replay` 选择 fold 或单独 forward；
- `swa_gates_prefix_match=False`：只要池上有 request window，SWA 不参与前缀
  匹配（`mem_cache/kv_cache_builder.py:396-405`）；
- 请求校验：拒绝 video/audio/input_embeds/positional override，image 仅 CUDA
  （`tokenizer_manager.py:1377-1393`，`schedule_batch.py:2351-2371`）；
  预热只跑 text（`http_server.py:2415-2420`）。

### 2.6 启动解析与兼容矩阵

`arg_groups/deepseek_v4_hook.py:333-351, 374-414, 424-436`：

- 硬件：CUDA 或 gfx950；
- prefill graph：`DISABLED` 或 `BREAKABLE`；ROCm 上不允许 prefill graph；
- 不允许：CP、DP attention（ROCm）、unified memory、PD decode、
  PD prefill（ROCm）、mixed chunk、LoRA、session radix；
- V4.1 全局不支持：非 DSpark 投机、HiSparse、unified-KV triton、
  TRT-LLM + encoder replay、two-batch overlap、PP>1；
- chunk 预算：`chunked_prefill_size >= 128 + page_size`，
  `max_running_requests > 0`（`:333-351`）。

### 2.7 PD

- P 角色支持（CUDA）；D 角色拒绝（`deepseek_v4_hook.py:399-405`）；
- 窗口状态通过 `SWA_RING` 走 PD 传输
  （`mem_cache/deepseek_v4_memory_pool.py:1687-1700`），
  `swa_valid_from` 标记请求已写的窗口起点。

---

## 3. 扩展点清单（NPU 适配的接缝）

1. **平台能力谓词**：`SRTPlatform.supports_*` / 图工厂
   （先例 `platforms/interface.py:197` `supports_speculative_algorithm`；
   `docs/graph/02_dspark_npu_accept_verify_graph_plan.md` 明确选择实例方法
   而非 `PlatformCapabilities` 字段）。
2. **后端虚函数**：`late_layer_num_tokens`、`tail_forward_metadata`、
   `write_tail_source_kv`、`enter/exit_late_layer_tail`、
   `folds_encoder_swa_replay`、`get_swa_out_cache_loc`。
3. **模型循环**：`deepseek_v4.py:4160-4333` 已平台无关，
   `_is_cuda` 分支只包住 `residual_batch`。
4. **DP helpers**：`models/deepseek_common/decoder_swa_bounded_replay.py`
   纯 torch + gloo `cpu_group`，NPU 可用。
5. **行选择/布局**：`late_layer_tail_layout` / `late_layer_tail_lens`
   是设备无关 Python（只用 `async_h2d`），但位于 Triton 模块内，
   NPU 需要 Triton-free 的导入路径。
6. **请求窗口**：`WindowLayout`/`window_layout` 有纯 torch fallback；
   复制/提交 kernels 只有 Triton；`RequestWindowSpec` 依赖 `KVLayout` 字节寻址。
7. **图回放**：`RowBucketGraphRunner` 基于 CUDA BCG；NPU 需要自己的
   `NPUGraph` 行桶实现，但 `DecoderReplayTail` 的 plan/DP/bucket 编排可复用。

NPU 当前实现对照（2026-11-19）：tail 布局/floor 与 fold/调度/radix 全部
复用本目录描述的 GPU 共享代码；request window 换成行式实现
（`NPURequestWindowController`），prefill/fold 用紧凑 staging，decode/verify
用 per-request ring 页表直接寻址。M2 行桶图已实现并完成 gpt_review2 P1
修复：图 key 加入 padded 请求数、`positions`/`input_ids` 图持有缓冲、DP
作用域成对进入/退出、DeepEP low-latency/容量门限、分页 SWA sink 用保留
页 0；decode/verify graph 已解锁。PD（P/D）在 NPU 上仍拒绝，待连接器
消费窗口页（见 `02`/`03`/`04`）。

---

## 4. 测试覆盖（GPU 现状）

| 测试 | 覆盖 |
|---|---|
| `test/registered/unit/model_executor/test_encoder_swa_replay.py` | fold 行映射、floor、logprob 行数、compressor 写 padding 槽、prefill graph padding、配置门、MLP-sync 计数 |
| `test/registered/unit/layers/test_dsv41_late_layer_tail_readers.py` | whole-extend 选择、layout、prefill 图容量/DP 投票一致、reader guard、DP 行交换时机、tail 内 low-ratio break |
| `test/registered/unit/model_executor/test_num_token_non_padded_localization.py:500-648` | tail 图只接受自有 MoE 计数域、bridge buffer 生命周期 |
| `test/registered/e2e/models/test_deepseek_v41_flash_gb300_trtllm_bounded_bcg.py` | 默认 decoder replay + TRT-LLM + BCG |
| `test/registered/e2e/models/test_deepseek_v41_flash_gb300_encoder_bcg.py` | encoder replay + breakable prefill |
| `python/sglang/test/dsv41_flash_common.py` | GSM8K/MMMU-Pro + DSpark accept 检查 |

NPU 侧目前没有任何 bounded replay 实现或测试。

---

## 5. 明确不在本设计范围

- NPU prefill CUDA/NPU graph 内的 tail 路径（NPU 默认无 prefill graph）；
- context parallel（GPU 本身也不支持与 decoder replay 组合）；
- encoder replay 的 PD decode 角色（GPU 也不支持）；
- 非 DSpark 投机解码。
