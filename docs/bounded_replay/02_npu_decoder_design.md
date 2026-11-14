# Decoder SWA Bounded Replay —— NPU 适配设计（A3 → A5）

- 版本：v1.2（2026-11-19，审查修复与回归）
- 基线：本仓库 `pr-49005-dsv41`（A3 DSV4.1 适配已合入）
- 对照基线：[01_gpu_bounded_replay_architecture.md](01_gpu_bounded_replay_architecture.md)
- 顺序：A3（arch22）M1 → M2，A5（arch35）M3
- 默认：NPU 上默认关闭，显式 `--enable-decoder-swa-bounded-replay` 开启

---

## 1. 目标与原则

- **复用 GPU 框架**：模型侧 tail 切换（`models/deepseek_v4.py:4160-4333`）、
  DP 行数交换（`forward_batch_info.py:1929-1942`）、tail 布局数学
  （`late_layer_tail_layout`）、`RowBucketGraphRunner` 的行桶/DP 编排全部复用。
- **能力谓词替代设备判断**：公共代码不新增 `is_npu`。
  新增 `SRTPlatform.supports_decoder_swa_bounded_replay(...)` 与
  `SRTPlatform.create_decoder_replay_tail(...)`；CUDA/HIP override 精确复刻
  现真值表，NPU override 返回 A3/A5 增量。
- **GPU 行为零变化**：公共改动只做"拆分/搬迁/注入"，并用单测锁死 CUDA 行为。
- **NPU 机制收敛**：全部落在
  `hardware_backend/npu/attention/ascend_dsv4_backend.py`、
  `hardware_backend/npu/dsv4/`、`hardware_backend/npu/graph_runner/`。
- **不新增环境变量**：复用 `SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS`；
  `..._MIN_PREFILL_TOKENS` 只对 CUDA prefill graph 有意义，NPU 上无效。

## 2. 现状差距矩阵

| # | 机制 | GPU 基线 | NPU 现状 | 影响 | 阶段 |
|---|---|---|---|---|---|
| 1 | 启动解析 | `deepseek_v4_hook.py:467-515`，CUDA/HIP 自动开启 | 显式开启会通过 resolver，随后崩溃；自动开启在 NPU 上为 False | 必须接平台谓词 | M1 |
| 2 | 模型 tail 切换 | `deepseek_v4.py:4160-4333` | 平台无关；`_is_cuda` 只包 `residual_batch` | 模型零改动 | M1 |
| 3 | tail 元数据 | `_build_late_layer_tail_metadata`（CUDA） | NPU 无 | 需 NPU 版 | M1 |
| 4 | source KV / index K | CUDA 压缩步内写 index K | NPU index K 在 index 步写（`_dsv41_index_k_store`） | 必须单独写全量 index K | M1 |
| 5 | `get_swa_out_cache_loc` | 有 tail 分支 | 无 tail 分支（`ascend_dsv4_backend.py:4623-4642`） | tail 行 KV 写错槽 | M1 |
| 6 | 行桶图 | `DecoderReplayTail` + `RowBucketGraphRunner`（CUDA BCG） | 只有全图 `NPUCudaGraphBackend`，无 BCG、无 prefill graph | 需要 NPU 行桶 runner | M2 |
| 7 | DP attention | 行数交换 + `enter_late_layer_dp`（SUM_LEN） | NPU MoE 后端对 padding 模式有额外约束 | 先拒绝，后按后端验证 | M1 拒绝 / M2b |
| 8 | DSpark | aux 按 tail 行捕获 | NPU DSpark 已支持，但未与 tail 联调 | 需真机验证 | M1 |
| 9 | Radix reprefill | 由 feature flag 驱动 | 自动继承（`UnifiedRadixCache`） | 需测试 | M1 |
| 10 | PD | CUDA P/D 都支持 | 未验证 | 先 P 后 D | M3+ |
| 11 | A5 | MQSMLA / fp8 行 | 现有 arch35 分支 | 单独验证 | M3 |

> 实现状态（2026-11-14）：M1 已落地并真机验证（含 DSpark）；
> DP attention 在 `attn_dp_size == tp_size` + DeepEP 下验证。
> M2 首版与 2026-11-19 审查修复见 §9/§9.1；A5 未落地。

## 3. 公共代码改动（最小集）

### 3.1 Triton-free 的 tail 模块

新增 `python/sglang/srt/layers/attention/dsv4/late_layer_tail.py`，从
`deepseek_v4_backend.py` 与 `attn_metadata_kernels.py` 搬迁：

- `SWA_WINDOW = 128`、`LateLayerTail`、`_tail_rows`、
  `_late_layer_whole_extend`、`late_layer_whole_extend`、
  `late_layer_num_tokens`、`_pad_rows`；
- `late_layer_tail_lens`、`late_layer_tail_layout`、
  `_whole_extend_tail_layout`。

只依赖 `torch` / `msgspec` / `async_h2d` / CP 元数据类型。原位置改为 re-export，
保持 `deepseek_v4_backend.py`、`deepseek_v4_backend_hip_radix.py`、
`deepseek_v4_replay_tail.py` 的导入不变，CUDA 行为零变化。

### 3.2 平台谓词与图工厂

`platforms/interface.py` 增加实例方法（默认 False / None）：

```python
def supports_decoder_swa_bounded_replay(self, *, model_type: str) -> bool:
    return False

def create_decoder_replay_tail(self, model):
    return None   # CUDA override 返回现有 DecoderReplayTail
```

- `platforms/cuda.py` / `platforms/rocm.py`：精确复刻现有真值表；
- `platforms/npu.py`：A3 先返回 `model_type == "deepseek_v41"`；
  A5 验证通过后再打开；
- `create_decoder_replay_tail` 的 NPU 实现延迟 import
  `hardware_backend/npu/dsv4/npu_decoder_replay_tail.py`。

### 3.3 注入 `DecoderReplayTail`

`models/deepseek_v4_replay_tail.py:155-173` 的 `maybe_create` 改为
`current_platform.create_decoder_replay_tail(model)`；现有 `is_cuda()`/env 门
原样搬进 CUDA 工厂。`DecoderReplayTail` 的 plan/DP/bucket 编排保持，
NPU 只替换 runner + layer-stack body。

> M2 首版已落地（2026-11-17）：`maybe_create` 走平台工厂，NPU 返回
> `NpuDecoderReplayTail`；M1 eager 路径保持不变。首版覆盖单请求/rank、
> 128 行桶与 DSpark aux；剩余见 §9。

### 3.4 Resolver

`arg_groups/deepseek_v4_hook.py:467-515`：

- 自动开启条件加上
  `current_platform.supports_decoder_swa_bounded_replay(...)`；
- 显式开启分支（`:489`）在 NPU 上同样执行兼容性检查，
  不满足时给出明确 `ValueError`（而不是后续崩溃）；
- NPU 默认值保持 `False`（`Optional[bool] = None` 时）；
- 新增 checkpoint 配置校验（对 CUDA 同样有益）：
  - `hf_config.kv_source_layer_ids` 非空；
  - `max(...) in index_source_layer_ids`；
  - `candidate_source_layer_id in (-1, max(...))`；
  - 最后一个 source 层 `compress_ratio in (1, 2)`（NPU 的
    `write_tail_source_kv` 需要 compressor）。

### 3.5 DP 约束

- M1：NPU resolver **拒绝 DP attention**（GPU 的 DP 路径依赖
  `enter_late_layer_dp` 强制 `SUM_LEN`，NPU MoE 后端未必接受；先保证 TP-only
  正确性）。
- M2b：先验证 `moe_a2a_backend == "none"`（TP-MoE gather 路径）在 2/4/8 卡上
  的 late-layer 行交换与 idle rank；通过后再评估 DeepEP 等后端。

## 4. M1 —— A3 eager bounded tail

目标：late layers 只跑 tail 行，eager 执行，先拿正确性与主要计算收益。

### 4.1 后端状态

`DeepseekV4AscendAttnBackend`（`ascend_dsv4_backend.py:2900`）：

- `self.enable_decoder_swa_bounded_replay`（来自 `get_exec().features`）；
- `self.tail_forward_metadata: Optional[ForwardMetadata] = None`；
- `self.tail_source_kv_layer_id: Optional[int] = None`
  （mixins 里已有类属性，补上实例生命周期与注释更新）。

### 4.2 元数据构造（关键重构）

`init_forward_metadata`（`:4175-4296`）拆成可重入的
`_build_npu_forward_metadata(forward_batch) -> ForwardMetadata`：

```python
def init_forward_metadata(self, forward_batch):
    self.forward_metadata = self._build_npu_forward_metadata(forward_batch)
    self.tail_forward_metadata = (
        self._build_late_layer_tail_metadata(forward_batch)
        if self.enable_decoder_swa_bounded_replay
        and forward_batch.forward_mode.is_extend_without_speculative()
        else None
    )
```

原因：`AscendAttnBackend.init_forward_metadata`（`ascend_backend.py:573-575`）
每次调用都会 `self.forward_metadata = ForwardMetadata()`；tail 构造必须
save/restore，并且不能通过递归 `init_forward_metadata` 破坏外层状态。

`_build_late_layer_tail_metadata` 的步骤：

1. 复用 `late_layer_tail_layout(...)` 得到 `token_indices`、`tail_lens_cpu`、
   `swa_replay_start`；`whole_extend = late_layer_whole_extend(..., dspark=
   self._is_dspark_algorithm)`；
2. 若 `encoder_swa_row_floor` 存在，按 CUDA 语义取 max（为 E1 预留）；
3. 构造 **tail ForwardBatch 视图**（`copy.copy` + 字段替换）：
   - 不变：`batch_size`、`req_pool_indices`、`seq_lens`、`seq_lens_cpu`、
     `forward_mode`、`spec_info`；
   - 替换：`positions`、`out_cache_loc`、`input_ids` 为 tail 行；
   - `extend_seq_lens(_cpu) = tail_lens`；
   - `extend_prefix_lens = seq_lens - tail_lens`
     （即每个请求的 tail 起点）；
   - `extend_start_loc = cumsum(tail_lens) - tail_lens`；
   - 单请求用 `contiguous_start` 切片，多请求用 gather（与 CUDA 一致）；
4. 保存 `self.forward_metadata`，调用 `_build_npu_forward_metadata(tail_view)`，
   恢复；得到 tail 元数据；
5. 附加 `LateLayerTail`：
   - `token_indices/positions/extend_seq_lens(_cpu)`；
   - `swa_out_cache_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
     tail_view.out_cache_loc)`；
   - `contiguous_start`；
6. 断言 `self._dsv4_sliding_window_size == SWA_WINDOW`；
7. 断言 tail 行数 > 0。

对 A3 的 `_build_seq_metadata_prefill`（`:438-474`）：
`start_pos = extend_prefix_lens = seq_len - tail_len`，`seqused = tail_lens`；
对 ratio-1 元数据（`:2451-2527`），`positions_cmp_padding_c1` 和 `c1_loc`
自然来自 tail 视图，索引 K/压缩器写入只作用于 tail 行（除了 source 层，
见 4.3）。

### 4.3 `write_tail_source_kv`

```python
def write_tail_source_kv(self, *, layer, x, positions, forward_batch):
    assert layer.compress_ratio in (1, 2) and layer.compressor is not None
    if x.shape[0] > 0:
        latent = self._low_ratio_compress(layer, x, positions, forward_batch)
        if layer.indexer is not None and layer.indexer.owns_k:
            self._dsv41_index_k_store(layer, latent)
    self.tail_source_kv_layer_id = layer.layer_id
```

契约：

- `x`/`positions` 是**完整 extend**（模型在 `deepseek_v4.py:4231-4249` 传入的是
  未裁剪的 `source_x` 与 `positions`）；
- **必须**同时写压缩 KV 和 index K：NPU 的 index K 在 index 步写
  （`_dsv41_index_k_store`，`:1493-1577`），若只跑
  `forward_low_ratio_sources(run_indexer=False)` 会静默丢失全量 index K
  （CUDA 在压缩步内写，所以没有这个问题）；
- 不跑 topk、不跑 attention。

配套 guard：`_low_ratio_index_topk` 在 `latent is None` 时，
若 `layer.layer_id == self.tail_source_kv_layer_id` 或
`forward_batch.encoder_swa_replay`，直接返回（K 已写/已缓存）；
其他情况保持原来的显式报错。

### 4.4 enter / exit

`enter_late_layer_tail(forward_batch) -> tuple` 保存：

- `self.forward_metadata`；
- `forward_batch.positions`、`out_cache_loc`、`input_ids`；
- `extend_seq_lens`、`extend_seq_lens_cpu`、`extend_prefix_lens`、
  `extend_prefix_lens_cpu`、`extend_start_loc`；
- `global_num_token_non_padded_cpu`、`global_num_token_non_padded`、
  `num_token_non_padded`。

然后：

- 把上述 batch 字段替换为 tail 视图的值；
- `self.forward_metadata = self.tail_forward_metadata`；
- M1 无定长块，不需要钳 `real_rows_count`；M2 按 CUDA
  `deepseek_v4_backend.py:2196-2208` 的方式钳。

`exit_late_layer_tail(saved, forward_batch)` 恢复全部字段，并清空
`tail_source_kv_layer_id`。

**为什么要换 `forward_batch` 字段而不只是换 metadata**：NPU 的
`_dsv41_window_indices`（`:2059-2091`）、`_dsv41_cmp_topk_length`、
`_build_dsv41_ratio_metadata`（`:2451-2527`）以及 A3 的 live mask
（`:2264-2268`）都在 attention 时刻直接读 `forward_batch.positions /
out_cache_loc`。只换 metadata 会让这些读者看到完整 extend 行。
`prepare_dsv4_cp_metadata`（`:3276-3281`）已有临时换 `positions` 的先例。

### 4.5 `get_swa_out_cache_loc`

在缓存检查前增加：

```python
tail = getattr(metadata, "late_layer_tail", None)
if tail is not None:
    return tail.swa_out_cache_loc
```

这样 `store_cache` 收到的 tail 行 K 会写到正确的 SWA 槽，而不是按完整
extend 的 `out_cache_loc` 翻译（长度不匹配）。

### 4.6 其他模型侧确认

- `late_layer_num_tokens(forward_batch)`：feature 关闭返回 `None`，
  否则调共享 helper，`dspark=self._is_dspark_algorithm`。
- `models/deepseek_v4.py` **不需要改动**：tail 切换、logits 裁剪、
  DSpark aux、scatter 已平台无关。
- `_low_ratio_sources_break`（`deepseek_v4.py:930-940`）依赖
  `is_in_breakable_cuda_graph()`，NPU 恒 False，因此 NPU 会直接调用
  `attn_backend.write_tail_source_kv`，符合预期。

### 4.7 M1 验收

- 长 prompt 贪心解码 on/off 前缀一致（允许 bounded replay 固有的数值差异，
  以 KL/精度阈值判定）；
- prompt logprob 请求与 full prefill 完全一致（whole-extend）；
- DSpark accept length > 1 且无明显回退；
- radix 前缀命中时 `swa_reprefill_tail_tokens` 生效；
- 性能：长 prompt prefill TTFT 相对 full prefill 明显下降。

## 5. M2 —— A3 NPU 尾层图回放

目标：把 late-layer stack 从 `torch.npu.NPUGraph` 回放，降低调度/launch 开销。

### 5.1 桶几何与图 key（实现状态）

实现按**总行数**取 128 对齐桶：

```text
bucket_rows = ceil(max_live_rows / SWA_WINDOW) * SWA_WINDOW
```

- 每请求真实行数 = `tail_lens[i]`，不足部分用 dummy 请求补齐，dead 行用
  dead-row 掩码屏蔽；live 行在前，dummy 行在 sink；
- **图 key 包含 padded 请求数**（`has_pre, dp, capture_dspark, bucket_rows,
  padded_bs`）：同一个行桶可以对应 `1×128` 或 `2×64`，两者的
  `req_pool_indices`/`extend_seq_lens` 张量形状不同，不能共用一张图；
- DP 下各 rank 在捕获前通过 `all_gather_single(tp_group.cpu_group)` 交换
  padded 请求数；全部一致才捕获/回放，不一致时**所有 rank 一致回退 eager**，
  避免捕获 barrier 与集合通信不对称；
- whole-extend 请求行数超过 `MAX_ROWS`、encoder request window、CP，以及
  MoE 不可捕获时回退 eager。

MoE 捕获门限（`_moe_can_capture`）：只接受 `none`、DeepEP v1/v2；DeepEP 必须
解析为 low-latency（显式 `--deepep-mode normal` 拒绝），且 `bucket_rows` 不超过
`SGLANG_DEEPEP(_V2)_NUM_MAX_DISPATCH_TOKENS_PER_RANK`；其他传输回退 eager。

### 5.2 新模块

`hardware_backend/npu/dsv4/npu_decoder_replay_tail.py`（或
`graph_runner/npu_tail_graph_runner.py`）：

- `NpuRowBucketGraphRunner`：
  - 每个 (bucket, key) 一张 `torch.npu.NPUGraph`；
  - 独立 graph pool，2 次 warmup 后捕获；
  - 图持有静态输入：HcState leaves、`positions`、`input_ids`、
    `input_ids_global`（后三者单独分配，**每次 replay 原地刷新**，避免同形状
    第二波读到捕获时的 token/位置）；
  - 首次使用/启动期捕获（建议启动期一次性捕获有限桶集合，
    `SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS` 限制；NPU 初始建议 512）；
  - replay 前原地刷新元数据，不重新分配张量；
  - padded batch 的 `seq_lens_cpu` 保持 **CPU tensor**（Ascend metadata 会调用
    `.int()`），`extend_seq_lens_cpu` 保持 list；
  - dead SWA 写槽：分页 target 池用保留页 0，draft ring 用 ring 外的备用行，
    request window 用自身 sink/workspace scratch；payload 在设备端清零
    （无 host sync）。
- `NpuDecoderReplayTail`：实现 `plan/run/run_idle`，
  `breaks_out_of_prefill_graph` 恒 False（NPU 无 prefill graph）。
- body：复刻非 CUDA 层循环（`deepseek_v4.py:4387-4393`），
  用 `layer.forward_hc_pre_from_prev(...)`；包含
  `dspark_layers_to_capture` 的 aux 捕获、expert-distribution recorder、
  最终 `state.materialized(...)`；断言 tail 内无 Engram。

### 5.3 元数据原地刷新清单

按 `_init_dsv4_graph_metadata`（`:3390-3540`）/
`_apply_dsv4_graph_metadata`（`:4104-4173`）的模式，为每个桶分配并在 replay
前刷新：

- `block_tables` / `block_tables_swa` / `swa_page_table`；
- `actual_seq_lengths_q_pa(_cpu)`、`actual_seq_lengths_kv`；
- `start_pos`、`seqused`；
- `swa_loc`（tail 写槽）；
- `positions_cmp_padding_c1`、`c1_loc`；
- `dsv41_win_sparse_indices` / `dsv41_win_topk_length` /
  `dsv41_cmp_topk_length_c1`；
- `kernel_metadata`（A3：固定 2048-int32 `win_metadata` / `c1a_metadata` /
  `c1a_qli_*`；A5：probe 尺寸）；
- `dsv41_live_mask_buf`；
- `num_token_non_padded` 设备标量（MoE top-k 用）。

约束：捕获后张量地址不可变；每步只允许原地写。回归测试要覆盖
"连续 replay 之间输入变化"。

### 5.4 DP / idle / DSpark

- M2a：TP-only 图回放；
- M2b：DP attention 图回放，每个 rank 回放最大 rank 的桶，
  idle rank 走 `run_idle`；padded 请求数经 TP CPU group 交换，一致才捕获；
- **DP 作用域成对**：捕获（warmup + capture）与每次 replay 都调用
  `enter_late_layer_dp()` / `exit_late_layer_dp()`，恢复 MoE 的 DP buffer
  几何；`_NpuTailGraph` 保存本图的 `dp_rows`；
- DSpark：图 key 含 `capture_dspark`；若 `mhc_stream_mean` 或 NPU MoE
  集合通信在 tail body 内不可图捕获，DSpark 场景回退 eager，并记录差距。

## 6. M3 —— A5（arch35）

- A5 验证通过后再打开平台谓词；
- 需要吸收的差异（详见
  `docs/read/15_PR49005_NPU_DSV4.1适配分析.md:213-230`）：
  - A3：arch22、BF16 行 1024 B/token、BF16 index K、`sparse_flash_mla`
    （长度 + mask）、per-rank query 头宽、固定 2048-int32 metadata ABI、
    纯 torch mHC；
  - A5：arch35、SWA 544 B FP8 / 压缩 320 B FP4、fp4 index K + e8m0 scale、
    MQSMLA 固定 64 query heads、probe 定长 metadata、融合 `npu_hc_pre_v2`、
    DSpark 稀疏索引为物理 SWA 槽。
- A5 当前对 TP 尾部 verify 行会报错，需确认是否影响 DSpark tail。
- 重新跑完整验证矩阵；按实测调整桶大小与 `MAX_ROWS`。

## 7. 风险与缓解

1. **index K 归属差异**：漏写 → 全 prompt 检索 K 丢失；guard 过宽 → encoder
   replay 覆盖缓存 K。缓解：`write_tail_source_kv` 同时写压缩 KV + index K；
   guard 只在 tail source / encoder replay 时跳过；单测覆盖 no-op 与 raise。
2. **位置驱动元数据**：任何未替换的 `forward_batch.positions/out_cache_loc`
   读者都会静默用全量行。缓解：tail 视图字段契约 + 单测 + enter/exit 形状断言
   + bs>1/非均匀 tail e2e。
3. **DP attention**：`enter_late_layer_dp` 强制 `SUM_LEN`，NPU MoE 后端可能
   要求对称/MAX_LEN 派发。缓解：M1 拒绝 DP；M2b 先验证 `moe none`，
   再逐后端放开。
4. **图新鲜度**：NPU 全图读固定地址，任何每步新建张量都会让图读到旧内存。
   缓解：固定缓冲 + 原地刷新 + 连续 replay 回归测试。
5. **DSpark / PD**：M1 已与 DSpark 联调通过（accept 3.3–4.7）；
   draft ring 的 `swa_valid_from` 与 encoder E3 联调通过。PD 仍未验证，
   分 P/D 两步验证，未通过前不宣称支持。

## 8. 实施顺序

| 步骤 | 内容 | 依赖 |
|---|---|---|
| T0 | Triton-free tail 模块 + re-export | — |
| T1 | 平台谓词 + resolver + checkpoint 校验 | T0 |
| T2 | M1 元数据重构、tail 视图、enter/exit、SWA 写槽 | T0/T1 |
| T3 | `write_tail_source_kv` + index-K guard | T2 |
| T4 | M1 单测 + A3 e2e（含 DSpark） | T2/T3 |
| T5 | M2 行桶图 runner + 策略注入 + 图/ea ger parity | T4 |
| T6 | M2b DP（`moe none` 先行） | T5 |
| T7 | A5（M3） | T6 |

## 9. 实现状态（2026-11-14）

开发目录：`/home/w00889861/dsv4/sglang_bouned_replay/sglang`
（分支 `pr-49005-dsv41`，基线 `6309017ada`）。

已落地：

- **T0**：新增 Triton-free 模块
  `layers/attention/dsv4/late_layer_tail.py`（`SWA_WINDOW`、`LateLayerTail`、
  tail 布局/whole-extend 判定、`apply_encoder_row_floor`）；
  `deepseek_v4_backend.py` 与 `attn_metadata_kernels.py` 保留 re-export，
  CUDA/HIP 导入路径不变。
- **T1**：`SRTPlatform.supports_decoder/encoder_swa_bounded_replay` 平台谓词；
  `decoder_replay_dp_moe_backends()` 与
  `validate_encoder_swa_bounded_replay_args(cfg)` 平台钩子；
  resolver 增加平台能力检查、显式开启校验、V4.1 source-layer 启动期校验，
  共享 resolver 不再新增 `is_npu` 分支。
- **M1**：NPU DSV4 后端的 tail 元数据构造（`_build_npu_forward_metadata`
  重入拆分 + tail ForwardBatch 视图）、`write_tail_source_kv`（压缩 KV +
  index K 全量写）、index-K guard、`late_layer_num_tokens`、
  `enter/exit_late_layer_tail`、tail SWA 写槽、窗口 floor
  （`dsv41_win_floor`，与 GPU 共用 `apply_encoder_row_floor`）。
- **M1 真机验证（A3，TP16/DP16 + DeepEP + eager decode）**：
  GSM8K 1.0000；DSpark accept 3.3–4.7；事实型 prompt 正确；
  平台 `supports_decoder_swa_bounded_replay` 在 A3 为真。
- **单测**：`test_npu_dsv41_decoder_tail.py`（21 例）、
  `test_bounded_replay_platform_gates.py`（11 例）、
  `test_dsv41_late_layer_tail_readers.py`（13）、
  `test_encoder_swa_replay.py`（18）等；`unit_sweep.sh` 13/14 文件全绿
  （1 个为基线已知失败）。

待办：

1. **M2**：首版已实现并在 A3 验证（2026-11-17/18）：平台图工厂
   `SRTPlatform.create_decoder_replay_tail` + `NpuDecoderReplayTail`；
   单请求/rank、128 行桶、DSpark aux 捕获、metadata 原地刷新、decode
   graph 共存均已跑通（两波 16/16 正确）。**DP padding/dead-row/idle
   rank 与短尾/whole-extend 的 padded batch 已实现**：按所有 rank 的
   最大 live 行数取桶，live 行在前，缺少的行用 `ReqToTokenPool` 预留
   slot 0 的 dummy request 补齐，dummy SWA 写槽映射到 sink 行；
   whole-extend 通过 padded batch 的 logprob 起始标记保留 plain
   window。A3 真机验证因外部租户占用 16 die 暂未执行；encoder 窗口
   组合与性能矩阵待补。
2. **decode/verify NPU graph（2026-11-17 已解锁）**：HCCP 初始化提前到
   第一个 AICPU metadata kernel 之前（`ensure_hccp_initialized`），A3
   无 replay 与 E3 target verify graph 均 capture 通过；resolver 不再
   要求 `--disable-cuda-graph`。DSpark draft ring 的 graph capture 仍走
   eager。
3. 完整 M1 场景矩阵：chunked prefill、prompt logprob、bs=8 非均匀长度、
   radix 命中（E1 已覆盖部分）。
4. **A5（M3）**：MQSMLA/QLI metadata、fp8/fp4 行格式、DSpark tail。

### 9.1 2026-11-19 审查修复（gpt_review2）

M2 的 5 个 P1 已修复（`npu_decoder_replay_tail.py` /
`dsv4_memory_pool.py`）：

1. **分页 SWA 写槽**：`set_swa_buffer` 不再把 dead 行写到 `pool.size`
   （最后一个可分配页的首行，真实 KV）。分页 target 池写保留页 0，draft
   ring 写 ring 外的备用行，request window 写自身 sink / workspace scratch；
   dead 行 payload 在设备端清零（无 host sync）。fused 入口的重复映射已删除，sentinel 由
   pool 统一处理。
2. **padded batch 类型**：`seq_lens_cpu` 恢复为 CPU tensor，避免 Ascend
   metadata `.int()` 报错；`extend_seq_lens_cpu` 仍为 list。
3. **图输入新鲜度**：`positions` / `input_ids` / `input_ids_global` 改为图
   持有缓冲，每次 replay 原地刷新；同桶、同形状、不同内容的第二波不再读
   捕获时的值。
4. **图 key**：加入 padded 请求数；DP 各 rank 用 TP CPU group 交换该计数，
   全部一致才捕获/回放，否则全体回退 eager；
   `_build_padded_tail` 断言构建值与 plan 交换值一致。
5. **MoE 捕获门限**：`_moe_can_capture` 只允许 `none` / DeepEP v1/v2
   low-latency，且桶不超过容量 env；显式 `normal`、超容量、其他传输回退
   eager。
6. **DP 作用域**：捕获与 replay 成对调用 `enter/exit_late_layer_dp`。
7. **D2H 同步**：`_tail_compress_rows` 用 fold extend 的 host 行数做 mask
   上界，去掉 `int(rows.max())`。

回归：新增 `test_npu_dsv41_tail_graph_guards.py`（5 例），并扩展
`test_npu_dsv41_decoder_tail.py`；`unit_sweep.sh` `NEW_FAILED_FILES=0`
（唯一失败为基线已知 `test_npu_dsv41_low_ratio_backend.py`）。M2 DP
padding/whole-extend 的两波真机对照仍待设备。
