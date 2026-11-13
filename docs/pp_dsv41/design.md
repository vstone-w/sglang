# DeepSeek-V4.1 NPU 平台 PP 详细设计

> 代码基线：`/home/w00889861/dsv4/sglang`，分支 `dsv41_pp`，已 rebase 到
> `upstream/main` `e183c415bd`（2026-11-20）。上游同日合入 DSV4.1 A5 支持
> （#47660），分支的 A5 系列以上游版本为准；本分支只在其上维护 A3、
> DSpark 与 PP 改动，公共 mHC `pre` 交接与切分校验仍由本分支提供。
> 分析背景与模型约束见同目录 `README.md`；本文描述**当前实现**的数据流、
> 不变量、失败模式与验收方式。

## 1. 能力边界

| 形态 | 状态 | 生效校验 |
|---|---|---|
| PP + 无投机 | 支持（NPU A3/A5；公共代码同样具备） | `validate_dsv41_pp_partition` |
| PP + EAGLE + `SGLANG_ENABLE_PP_SPEC=1` | 支持（PD 或非 PD；要求 `--disable-overlap-schedule`，禁止 adaptive / attention DP） | `validation_hook.check_pipeline_parallel_compat` |
| PP + EAGLE 无 PP spec relay | 仅 PD `disaggregation-mode=prefill` | 同上 |
| PP + DSpark，`disaggregation-mode=prefill` | 支持（P 端 PP，D 端 `pp_size=1`） | `speculative_hook._handle_dspark` |
| PP + DSpark，`disaggregation-mode=null/decode` | **拒绝** | `speculative_hook` / `validation_hook` |
| PP + DSpark + `SGLANG_ENABLE_PP_SPEC=1` | **拒绝** | `validation_hook` |
| PP + vision tower | **拒绝**，需 `--language-model-only` | `validate_dsv41_pp_partition` |
| PP + decoder SWA bounded replay | **拒绝** | `deepseek_v4_hook` 的 incompatible 列表 |
| PP + overlap schedule | **拒绝** | `check_pipeline_parallel_compat` |
| PP + `--min-free-slots-delay` | **拒绝** | 同上 |
| PP + attention DP 且开 PP spec relay | **拒绝** | 同上 |

表中“支持”指配置校验与代码路径已具备；实机验收状态见第 12、13 节。

非 PD 的 PP + 投机只支持 EAGLE 的 PP spec relay；DSpark 的 decode 验证链依赖
末级 tail draft，当前实现不维护该形态（评审 P0，见第 11 节）。

## 2. 数据流与分层

一次 PP forward 的参与方：

```
Scheduler (每个 pp_rank 一个进程)
  └─ PP event loop：mb_id 轮转、proxy 收发、microbatch 状态机
       └─ model_worker.forward_batch_generation(batch, pp_proxy_tensors=...)
            ├─ 非末级：TpModelWorker 只跑本 stage 的 target 层
            └─ 末级：DSpark/EAGLE worker 在 target 之上跑 draft / verify
```

- 层范围由 `get_pp_indices` / `compute_pp_indices` 统一计算，校验与运行时共用
  同一套算术（`distributed/utils.py`）。
- 每个 stage 的 KV 池只物化 `[start_layer, end_layer)`；PP 边界上的跨层共享状态
  （压缩 KV source、candidate、mHC `pre`）必须显式处理，见第 3、5 节。
- stage 之间的 forward 张量只通过 `PPProxyTensors` 传递：
  - `hidden_states`：末层输出，2D `[T, hc_mult * hidden_size]`（mHC）或
    `[T, hidden_size]`；
  - `mhc_pre`：fp32 `[T, hc_mult]`，仅 DSV4.1 的 lagged mHC 需要；
  - 既有可选键：`residual` / `attn_res_bank` / `topk_indices` /
    `dspark_hidden_states`（按模型与 spec 配置分配）。
- replay / eager / dummy 三条路径共用 `allocate_pp_proxy_tensors` 的键与 dtype；
  graph replay 按 key 原地拷贝，无需新增图逻辑。

## 3. mHC `pre` 跨 stage 交接

### 3.1 状态语义

DSV4.1 的 `hc_pre_from_prev_sublayer=True`：每个子层的 pre-mix 系数由**上一个
子层**产生，被**当前子层**的 combine 消费。跨 stage 时只交接这一项：

| 字段 | 形状 | dtype | 说明 |
|---|---|---|---|
| `hidden_states` | `[T, hc_mult * hidden]` | 模型 dtype | 3D mHC 张量 flatten 后走 IPC |
| `mhc_pre` | `[T, hc_mult]` | **fp32** | `HcState.pre`，token-major |
| `input` / `stats` | — | — | 层内状态，不跨层 |

`hc_mult=4` 时新增载荷为每条边每 token 16 字节。

### 3.2 实现路径

`DeepseekV4Model._forward_impl`（`models/deepseek_v4.py`）：

1. **first rank**：`embed_tokens` 后经 `hc_broadcast` 扩成 3D
   `[T, hc_mult, hidden]`。
2. **非 first rank**：
   - 从 `pp_proxy_tensors["hidden_states"]` 取 2D 张量并 view 回 3D；
   - `prev_pre = pp_proxy_tensors.tensors.get("mhc_pre")`；
   - `pp_size > 1` 且缺失时抛
     `mHC pre-mix hand-off across PP is missing`；
   - 校验 dtype 为 fp32、形状为 `(T, hc_mult)`，否则抛
     `mHC pre-mix has the wrong layout`。
3. `_forward_layers_hc_pre_from_prev(..., prev_pre=...)` 用
   `mhc.HcState(hidden_states, pre=prev_pre)` 初始化，逐层传递；末层产出
   `last_pre`。
4. **非 last rank**：返回
   `PPProxyTensors({"hidden_states": hidden_states.flatten(1), "mhc_pre": last_pre})`；
   若 `hc_pre_from_prev_sublayer` 却没有 `last_pre`，直接 assert。
5. **last rank**：`mhc_combine(pre_hc_head, last_pre, hc_mult, dtype)` 得到最终
   hidden states，进入 `lm_head` / logits。

CUDA 路径在 stage 边界关闭跨层融合：非 last rank 的最后一层传
`allow_cross_layer_fusion=False`，先把本 stage 的 post materialize，再 relay。
NPU 走 `NonCudaDecoderMixin.forward_hc_pre_from_prev`，直接消费 `state.pre`，
不需要额外分支。

## 4. PP proxy 缓冲区

宽度解析（`model_runner_components/misc_utils.py`）：

```python
resolve_pp_proxy_hc_pre_width(model_config):
    pp_size > 1 and pp_rank > 0 and hc_pre_from_prev_sublayer
        -> hc_mult
        else 0
```

分配（`runner_utils/buffers.py::allocate_pp_proxy_tensors`）：

- `mhc_pre`：`[max_hidden_tokens, pp_proxy_mhc_pre_width]`，fp32；
- 有效行数 = 本步 hidden token 数；dummy / graph 捕获按各自的 token 上界分配；
- 需要与 `hidden_states` 使用同一 `max_hidden_tokens`，否则 relay 端形状不一致。

四条透传路径：

| 路径 | 位置 |
|---|---|
| eager | `runner/base_runner.py`（`allocate_pp_proxy_tensors`） |
| decode graph | `runner/decode_cuda_graph_runner.py` 传宽度 |
| prefill graph | `runner/prefill_cuda_graph_runner.py` 传宽度 |
| dynamic chunk profiling | `scheduler_components/dynamic_chunk_sizer.py` 的 dummy proxy |

不变量：所有 rank 的 proxy 键集合与形状必须一致；`mhc_pre` 只在
`pp_rank > 0` 存在。graph replay 的按 key 拷贝会原地刷新 `mhc_pre`，前提是
`prepare_replay` 不走跳过 `load_batch` 的快路径（见第 7 节）。

## 5. 层切分校验

### 5.1 约束来源

对每个 stage `[start, end)`：

1. **压缩 KV source 组**：ratio 1/2 的层读“最近的、同 ratio 的 kv_source 层”。
   若 layer `L` 的最近同 ratio source `S` 不在同一 stage，KV 池构造/访问失败。
   注意：`S` 与 `L` 之间允许夹着其它 ratio 的层，运行时仍会跳过它们继续找
   `S`。因此**被阻塞的边界是 `(S, L_max]`**，其中 `L_max` 是下一个同 ratio
   source 之前最后一个同 ratio 层。
2. **index source 组**：低 ratio 层的 top-k 由最近的、同 ratio 的
   `index_source_layer_ids` 层产生，写入 per-forward 的
   `fm.c{ratio}_topk_indices`；后续层跨 stage 读不到。约束与 KV source 相同，
   但两组 source 不必重合（评审反例见 §5.3）。
3. **candidate source**：`candidate_source_layer_id` 与所有后续消费者必须同
   stage，实际效果是所有 stage 边界不得大于 candidate（candidate 及之后整段
   落在 last stage）。
4. **DSpark capture 层**：`dspark_target_layer_ids` 必须在 last stage；等价于
   last stage 的起点不得大于最早 capture 层。该字段缺失时参数校验跳过（运行
   时以 draft 配置解析出的捕获层为准，见 §5.5）。
5. **vision**：checkpoint 带 vision tower 时必须 `--language-model-only`。

### 5.2 算法

`legal_split_points(hf_config, num_layers, is_dspark=...)` 返回所有合法边界
（含 0 与 `num_layers`）：

```text
blocked = {}
for sources in (kv_source_layer_ids, index_source_layer_ids):
    for ratio in (1, 2):
        for each source S of this ratio:
            stop = next source of this ratio, or num_layers
            consumers = [L in (S, stop) with ratios[L] == ratio]
            if consumers: blocked += (S, consumers[-1]]
candidate = candidate_source_layer_id
if 0 <= candidate < num_layers: blocked += (candidate, num_layers)
if is_dspark and capture_layers:
    blocked += (min(capture_layers), num_layers)
legal = [0..num_layers] - blocked
```

`suggested_partition(hf_config, pp_size, is_dspark=...)` 在同一 `legal` 集合上做
有界 DP，最小化 `(最大 stage 层数, 各 stage 层数平方和)`，返回
`SGLANG_PP_LAYER_PARTITION` 字符串；无解返回空串。校验与建议共用同一个
blocked 集合，因此建议结果必然能通过校验。

### 5.3 示例

- 真实 checkpoint（40 层，source `[2,8,14,20]`，candidate 20，capture
  `[37,38,39]`）：合法边界 `{1,2,8,14,20,40}`，与 README §2.5 一致。
- 交错 ratio：`[0,2,0,2]`、source 1。layer 3 仍消费 source 1，边界 2/3 非法；
  合法边界 `[0,1,4]`，PP2 建议 `1,3`（旧实现会错误建议 `2,2`）。
- 40 层、capture 层 5：PP2 建议 `5,35`（旧实现建议 `20,20`，已修复）。
- KV/index source 分离（评审反例）：`[0,2,2,2]`，kv source `[1,2]`，index
  source `[1]`。旧实现只看 kv source，放行 `2,2`；stage 1 缺 index source，
  top-k 读不到。现合法边界 `[0,1,4]`，`2,2` 被拒绝，PP2 为 `1,3`。

### 5.4 失败信息

`_pp_split_error` 输出：非法 partition、具体约束（kv/index source/candidate/capture/
vision）、合法边界列表、`--pp-size` 下的建议 partition。校验失败发生在参数解析
阶段，无 torch/设备依赖。

### 5.5 运行时捕获层校验

参数校验读目标 config 的 `dspark_target_layer_ids`（可能缺失，缺失时跳过）。
运行时以 draft 配置解析出的捕获层为准（内置 draft 与
`--speculative-draft-model-path` 都适用），在 `SpecAuxHiddenStateConfig` 解析
完成后用 `validate_pp_capture_layers` 按实际 partition 确认所有捕获层都在
last stage；目标与 draft 配置不一致时在此处报错，而不是等到 graph capture
静默缺少 aux hidden。

## 6. NPU attention backend 的 stage 化

`DeepseekV4AscendAttnBackend.__init__` 从 KV 池的 `stage_layer_ids` 派生：

```python
stage_ratios = tuple(sorted({pool.compression_ratios[l] for l in pool.stage_layer_ids}))
self._dsv4_stage_layer_ids = frozenset(pool.stage_layer_ids)
self._dsv4_unique_compress_ratios = list(stage_ratios)
self._dsv4_has_c4 / _dsv4_has_c128 = 4 in stage_ratios / 128 in stage_ratios
self._dsv4_has_low_ratio = 模型为 V4.1 且非 draft worker
```

- 分配哪些 metadata / graph buffer 由 **stage ratios** 决定；某 stage 没有
  ratio 4/128/1/2 时不再访问空池。
- graph buffer 按本 stage 的 ratio 独立分配：SWA 的 `c1a` metadata 对每个 V4
  stage 都存在（包括全 ratio-0 stage）；`c4`/`c128` 的 page table、loc、
  padding、`c4a`/`c128a` metadata、`c4_topk_indices` 分别随
  `_dsv4_has_c4` / `_dsv4_has_c128` 分配与刷新，只有 C128 的 stage 也持有
  自己的缓冲。
- `_dsv41_low_ratios()` 返回本 stage 的 ratio 1/2 子集；compressor / indexer
  的循环只遍历它。
- `_dsv4_has_low_ratio` 保持**模型全局**：它同时是“是否走 DSV4.1 ds41 算子族”
  的信号，ratio-0 层也依赖它。
- candidate 消费者判断使用 stage 层集合：
  `_dsv41_has_candidate_consumers()` 只在本 stage 存在 candidate 之后的
  index source 时返回 True；切分校验保证 candidate 与其消费者同 stage。
- A3 / A5 算子族按 `is_npu_arch35()` 选择；A5 的 64 query-head 约束在启动时
  断言，不再提供跳过开关。算子包装（DSL compressor/indexer、`mhc_pre/post`、
  `inplace_partial_rotary_mul`、`moe_gating_top_k`、`kv_compress_epilog_v2`）
  由与 CANN 9.2 匹配的 `sgl_kernel_npu` wheel 提供。
- verify 模式：DSV4.1 只接受 STATIC 与 CAP_ACCEPT（两者共用 STATIC 布局），
  COMPACT 在启动时拒绝；错误信息给出 `static` / `cap-accept` 两种取值。

## 7. NPU graph 与 PP 通信

### 7.1 replay 路径

`NPUCudaGraphBackend.prepare_replay` 在检测到 `runner.buffers.pp_proxy_tensors`
非空时直接返回 False：

- pre-planned 快路径只刷新 input_ids/positions 等，不执行 `load_batch` 的 PP
  准备；
- 走通用 `load_batch` 路径可保证 `hidden_states` / `mhc_pre` 在图 replay 前
  按本轮 token 行数刷新。
- 该限制只影响 PP；`pp_size=1` 保留原有 pre-planned 路径。

### 7.2 通信

- NPU `pp_size == 2`：输出张量的 send 与配对 recv 合并为一次
  `batch_isend_irecv`，避免 HCCL 上分开 isend/irecv 互等。
- `pp_size > 2`：使用通用的 rank parity 顺序（偶数 rank 先 send、奇数 rank
  先 recv），避免多 microbatch 下环形死锁。
- HCCP 在第一个自定义 AICPU metadata kernel 之前完成 warm-up
  （`NPUCudaGraphBackend.__init__` 的 all-reduce），避免 CANN 版本相关的
  507018 失败。
- A3 的 metadata 种子、图内 dispatch 表等 NPU 特例全部留在 NPU backend /
  graph runner 内，公共 PP 循环只保留通用逻辑。

## 8. PD：P 端 PP + DSpark，D 端 pp_size=1

### 8.1 形态

- P：`--disaggregation-mode prefill --pp-size > 1 --speculative-algorithm DSPARK`，
  `SGLANG_ENABLE_PP_SPEC=0`。
- D：现有 DSpark decode 配置，`pp_size=1`。
- draft 只在 P 的最后一个 PP stage 构造（`_hosts_draft = pp_group.is_last_rank`），
  其余 stage 只跑 target；P 上完成 target forward、aux hidden 捕获与 draft KV
  注入，D 端沿用现有 DSpark PD decode 流程。
- 非 last stage 不跑 draft，也不做 verify；它们只在 prefill forward 中贡献本
  stage 的 target 层与 KV。

### 8.2 KV 布局自描述

PP stage 与 PP=1 decode peer 的层集合不同，位置对齐会错层。当前实现让 KV 池
描述自身布局：

| 接口 | 作用 |
|---|---|
| `get_contiguous_buf_infos()` | 按 **buffer 类型**顺序（ratio 组、KV/index-K/index-scale）返回 ptr/len/item_len |
| `get_contiguous_buf_layer_ids()` | 与上表 1:1 的全局 layer id；缺 ratio 的 stage 不再错位 |
| `get_request_state_layer_ids()` | request-scoped state（ratio-2 source 等）的全局 layer id |
| `get_state_layer_ids()` | SWA 组件（`get_state_buf_infos`）的全局 layer id，只含 SWA 行 |
| `get_c4_state_layer_ids()` | C4 state 组件（`get_c4_state_buf_infos`）的全局 layer id |

SWA 与 C4 state 是两个独立组件，id 必须各自与自己的 pointer 列表 1:1；把 C4
id 拼进 SWA 列表会让配对函数以“4 个 id 对 2 个指针”拒绝 A3 的 P 端 stage。

`Region.layer_ids` 与 state entries 复用这些 id；公共发送器通过
`build_transfer_entry_pairs(src_ids, dst_ids, n_src, n_dst,
allow_positional_fallback=pp_size == 1)` 按 layer id 配对：

- PP 下不做位置回退，双方缺 id 直接报错；
- repeat 的 layer id（K/V、多 state tensor）按出现顺序配对；
- decode peer 缺少某层的 entry 时报错。

PP stage 若某个组件一条 entry 都没有（例如所有层都是 ratio 0），仍会发布**空
占位**，保持 wire order 的位置稳定（`static_kv_storage._model_has_request_state_outside_stage`、
`prefill._request_state_payload`、`mooncake/conn.py` 的空 state 跳过）。

### 8.3 握手字段

`get_dsv41_spec_layout` 在 `pp_size > 1` 时只返回 **rank 一致**的字段：

```text
{"num_draft_tokens": ..., "compression_ratios": [...]}
```

并要求本 rank 至少有一个 target SWA state。per-rank 的层/state 字段不参与
跨 rank 比较（`common/conn.py` 的 `layout_keys`）；PP=1 仍比较完整字典。
这样 P 端各 stage 的 draft SWA / request state 差异不会误报 layout mismatch。

### 8.4 目的端步长校验

发送端用源 `item_len` 计算目的地址，因此**源/目的步长必须逐项相等**：

- 注册期：`_kv_layout_mismatch` 在双方都提供 `item_lens` 时逐项调用
  `_state_stride_mismatch`；有 layer id 的一侧按 id 配对，否则按位置切片。
  PP peer 不再被要求必须提供 ids/lens：V4.1 的位置配对不需要目的端 id。
- 发送期：hybrid MLA 的 PP 路径在 `_send_kvcache_generic` 内按 id 配对，步长
  不一致时终止该传输。DSV4.1 的低 ratio 层走位置路径，按 source 组切目的端，
  不读取目的端 ids/lens。
- 零条目 stage 表示零贡献；跨版本/异构部署必须双方同时升级。

## 9. PP 容量规划

每个 stage 独立计算 `bytes_per_full_token`，再由
`kv_cache_configurator._apply_token_constraints` 对 PP 各 rank 做 MIN，得到整条
流水线的 token cap，然后每个 rank 用该 cap 重新推导 pool 尺寸
（`calculate_pool_sizes_from_max_tokens`）。

全 ratio-0 stage 的 `bytes_per_full_token == 0`，本地预算无法约束 token 数：

- 先检查固定池（SWA cap / state / request window）是否超过 `available_bytes`；
  超预算直接报错，不再返回正容量；
- 未超预算时返回 `_NO_LOCAL_TOKEN_LIMIT`（`1 << 62`）哨兵；PP MIN 会忽略它，
  由真正有 token KV 的 stage 决定容量；
- 不再使用单请求 `context_len` 兜底，避免 PP4 `2,6,6,20` 这类首段全 ratio-0
  的切分被错误压低整条流水线；
- `max_tokens <= config.max_total_num_tokens` 的断言仍然成立；MIN 变小后才
  重新推导 allocator 尺寸。

## 10. 配置校验汇总

| 校验 | 位置 | 关键行为 |
|---|---|---|
| PP 切分 | `deepseek_v4_hook.validate_dsv41_pp_partition` | kv/index source 消费区间、candidate、DSpark capture、vision、建议合法切分 |
| DSPARK + PP | `speculative_hook._handle_dspark` | 仅 `disaggregation_mode=prefill` |
| DSPARK + PP + PP_SPEC | `validation_hook.check_pipeline_parallel_compat` | 拒绝 |
| EAGLE + PP_SPEC | 同上 | 要求非 adaptive、非 attention DP；无 relay 时仅 prefill |
| bounded replay | `deepseek_v4_hook` | PP 下拒绝 decoder replay |
| NPU verify 模式 | `DeepseekV4AscendAttnBackend._validate_spec_support` | V4.1 只允许 DSpark + STATIC/CAP_ACCEPT |
| A5 head 数 | `DeepseekV4AscendAttnBackend.__init__` | low-ratio 时断言 64 query heads |

## 11. 2026-11-19 评审结论落地

| 评审项 | 结论 | 处理 |
|---|---|---|
| P0 默认 wheel 缺算子包装 | 属实 | 部署要求 CANN 9.2 + 匹配 wheel；代码保持 DSL 包装，未加运行时回退 |
| P0 非 PD DSpark + PP 丢 verify proxy、tail 状态单槽位 | 属实 | 恢复 DSpark + PP 仅限 PD prefill，撤回 tail-draft/relay/verify 重建 |
| P1 PD prefill 多跑 tail draft / idle collective | 属实 | 随 P0 移除；保留 idle target 所需的 proxy 透传 |
| P1 cap-accept 被误拒、A5 头数诊断开关 | 属实 | 只拒绝 COMPACT；删除 `SGLANG_DSV4_ALLOW_ATTN_TP_SHARD` |
| P1 平台门控扩大 | 部分属实 | 不采纳平台门控：公共 PP 是本分支设计目标，加平台判定会扩大公共代码并新增 `is_npu` 分支 |
| P1 PD 异构 PP 步长校验 | 属实 | Ascend 发送端读取 decode 端 item lens 并逐项校验；非空 stage 缺元数据直接报错 |
| P1 全 ratio-0 stage 容量兜底 | 属实 | 固定池预算检查 + 无本地上限哨兵，PP MIN 后统一推导 |
| P2 切分建议可能非法 | 属实 | 修正 source 消费区间并纳入 DSpark capture 约束；建议与校验共用约束集 |
| 设计项：mHC 宽度 `start_layer` 未使用 | 属实 | 删除该参数 |
| 设计项：临时 PP trace / 堆栈打印 | 属实 | 从公共路径移除；graph 特例与 HCCP warm-up 本就在 NPU backend |

### 11.1 2026-11-20 gpt_review2 结论落地

| 评审项 | 结论 | 处理 |
|---|---|---|
| P1-1 用实际 draft 捕获层校验切分 | 属实 | 参数期容忍目标 config 缺 `dspark_target_layer_ids`；运行期由 `SpecAuxHiddenStateConfig` 解析出的层号经 `validate_pp_capture_layers` 按实际 partition 校验，覆盖内置/外置 draft |
| P1-2 index source 纳入切分约束 | 属实 | 合法边界同时按 `kv_source_layer_ids` 与 `index_source_layer_ids` 的消费区间计算；校验与建议共用；新增 KV/index source 分离用例 |
| P1-3 A3 SWA 层 ID 注册 | 属实 | SWA 组件只发布自己的 id；C4 state 由 `get_c4_state_layer_ids` 独立发布，id 与 pointer 1:1 |
| P1-4 只有 C128 的 stage 图缓冲 | 属实 | c1a（SWA）对每个 V4 stage 独立分配；c4/c128 的页表、loc、padding、dispatch 表分别随 `_dsv4_has_c4`/`_dsv4_has_c128` 分配与刷新 |
| P1-5 PD 布局错误限定到当前请求 | 已调整 | 注册期 `_state_stride_mismatch` 在双方提供 lens 时校验；V4.1 低 ratio 改为位置配对后不再要求 ids/lens，原 `pp_peer_kv_metadata_mismatch` 与发送期透传已移除 |
| 切分建议次目标（平方和） | 不采纳 | 两个合法解均可；现有 `(最长 stage, 平方和)` 是文档化的取舍，不移除 |
| 公共代码平台/后端能力门控 | 不采纳 | 与 2026-11-19 结论一致：公共 PP 是设计目标，加平台判定会扩大公共代码并新增 `is_npu` 分支 |
| 拆分编译/权重修复到独立提交 | 不采纳（当前分支） | 按用户要求保持单笔 PP 提交；编译守卫与配套修复的验证已分别记录 |
| draft 通信 dtype 改为 dispatcher 配置 | 暂缓 | 现实现以 `envs.override` 限定在 draft 执行期，已在 A3 实测；改 dispatcher 接缝需真机回归 |
| bounded replay 合入 | 部分已完成 | `83ae734` 不在本仓库；PP 与 decoder replay 的互斥在 rebase 后已由上游 hook 保留 |
| graph 回退覆盖首级 | 保留 | 首级不消费 proxy 输入，但快路径需真机 graph 验证后再收窄；当前保守回退正确性优先 |
| candidate metadata 按 stage | 已完成 | `_dsv41_has_candidate_consumers()` 已用 stage 层集合判定 |

## 12. 测试与验收

### 12.1 已执行的单元测试

```bash
cd /home/w00889861/dsv4/sglang
PYTHONPATH=$PWD/python:/usr/local/Ascend/cann-9.2.0/python/site-packages \
  python3 test/registered/unit/npu/test_dsv41_pp_partition.py
```

- `test_dsv41_pp_partition.py`：16 项通过（合法边界、文档 partition、非法切分、
  kv/index source 分离、缺失 DSpark 字段、运行时捕获层、PP peer 元数据）。
- `test_npu_ascend_dsv4_backend.py`：68 项通过（含 SWA/C4 id 1:1）。
- `test_npu_dsv41_low_ratio_backend.py`：102 项通过。
- `test_dsv41_dspark_metadata_rebase.py`：19 项通过（含全 ratio-0 / 仅 C4 / 仅
  C128 stage 的图缓冲与 capture）。
- `test_npu_cudagraph_backend.py`：6 项通过。
- A3 / DSpark / Engram / graph runner 等全部 NPU 单测通过；唯一例外
  `test_npu_seed_trace_rules.py` 的 2 项在 rebase 前的分支上以同一 torch
  构建同样失败（Dynamo 规则缓存冲突），非本次改动引入。
- 所有改动文件 `py_compile`、`git diff --check` 通过。

### 12.2 实机验收顺序

1. **A3 镜像与依赖**：确认 `sgl_kernel_npu` 提供 `compressor_v2`、DSL
   attention/indexer、`kv_compress_epilog_v2`、`mhc_pre/post_dpskv4`、
   `inplace_partial_rotary_mul`、`moe_gating_top_k`。
2. **普通 PP（无 spec）**：PP1/2/3/4，先 `--disable-cuda-graph`，再开 decode
   graph。检查 `mhc_pre` 的 shape/dtype、短序列、32K/64K、chunked prefill，
   与 PP=1 对照。
3. **PD：P PP2/4 + DSpark，D PP1 + DSpark**：prefix hit/miss、ratio-2 奇偶
   边界、ratio-0 stage、末级 draft SWA、交错 microbatch、忙闲 DP rank；核对
   KV 层映射、输出与 DSpark accept 长度。
4. **容量与回归**：PP4 `8,6,6,20` 与 `2,6,6,26` 的可用 KV token 数、TTFT
   P50/P95、prefill 吞吐；确认全 ratio-0 stage 不压低流水线容量。

## 13. 风险与后续

1. **GPU PP 未实测**：公共 mHC 交接与切分校验在 CUDA 上同样生效，但本分支只在
   NPU 验证；合入前需要 GPU PP 冒烟。
2. **A5 未完整验收**：A3 优先；A5 需单独验证 `mhc_pre/post`、共享 embed 与
   DSpark draft 路径。
3. **PP > 2 通信**：只有 PP2 使用 `batch_isend_irecv` 快路径，PP>2 走 parity
   顺序，需实测多 microbatch 不 hang。
4. **数值一致性**：PP 边界使跨层 mHC 融合退化为普通 post，可能与 PP=1 存在
   非 bitwise 差异；用 KL/GSM8K 对照。
5. **非 PD PP + DSpark 不支持**：若后续需要，必须复用 microbatch 槽位与按 RID
   重排，明确 proposal / verify window / KV 生命周期，并补 verify proxy 透传。
6. **算子版本耦合**：DSL 包装与 bundled binding 依赖 CANN 9.2 与匹配 wheel；
   跨版本部署需先跑导入与首轮 forward。

## 14. 参考

- 模型约束与结论：`docs/pp_dsv41/README.md`
- 历史 PR：`https://github.com/sgl-project/sglang/pull/39820`
- 真实权重：`/mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8`
- 关键实现：
  - mHC：`models/deepseek_v4.py`、`models/deepseek_v4_mhc.py`
  - 切分校验：`arg_groups/deepseek_v4_hook.py`
  - KV 池 / PD 布局：`mem_cache/deepseek_v4_memory_pool.py`、
    `hardware_backend/npu/dsv4/dsv4_memory_pool.py`、
    `mem_cache/static_kv_storage.py`
  - NPU attention：`hardware_backend/npu/attention/ascend_dsv4_backend.py`
  - 单测：`test/registered/unit/npu/test_dsv41_pp_partition.py`
