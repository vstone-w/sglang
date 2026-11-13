# DeepSeek-V4.1 PP 修改点与涉及文件

> 代码基线：`/home/w00889861/dsv4/sglang`，分支 `dsv41_pp`，HEAD
> `6d155e08d8`，已 rebase 到 `upstream/main` `e183c415bd`（2026-11-20）。
>
> 三份文档的分工：[`README.md`](README.md) 讲模型约束、能力边界与部署方式；
> [`design.md`](design.md) 讲数据流、算法、不变量与评审结论；本文逐条列出
> **PP 改了什么、为什么、在哪个文件**，供评审与后续维护定位。
> 行号对应当前工作区，rebase 后会漂移，以函数名为主索引。

## 1. 分类约定

- **PP 专属**：为支持 PP 新增或修改的逻辑；去掉 PP 就不再需要。
- **stage 化**：原本按模型全局计算、PP 下必须按本 stage 收窄的逻辑。
  语义不变，只是作用域从全模型缩小到 `[start_layer, end_layer)`。
- **配套（非 PP）**：同一分支为 A3 / DSpark 引入的修复。PP 场景会用到，
  但不是 PP 逻辑；评审与问题定位时需要和 PP 改动分开看。
- **上游基座**：本分支未修改、但 PP 依赖的上游实现（第 9 节），列出以免
  把上游行为误判成本分支改动。

## 2. 公共代码（跨平台）

### 2.1 mHC `pre` 跨 stage 交接【PP 专属】

根因：DSV4.1 的 `hc_pre_from_prev_sublayer=True`，每个子层的 pre-mix 系数由
**上一个子层**产生、被**当前子层**的 combine 消费。PP 只传 `hidden_states`
时，跨 stage 的 `pre` 没有承载者；若沿用上游的 `None` 兜底，首个 combine 会
退化为 one-hot stream 0（不报错但算错），因此分支改为在缺失时显式抛错。

`HcState.pre` 字段、lagged 语义，以及 `_forward_layers_hc_pre_from_prev`
返回 `state.pre`、last rank 用 `mhc_combine(pre_hc_head, last_pre, ...)` 收尾，
都是**上游已有**（`models/deepseek_v4_mhc.py:196-250`；`models/deepseek_v4.py`
上游的 `return state.residual, state.pre, tail` 与 `mhc_combine` 块）。
本分支新增的只是**跨 stage 的搬运**；`models/deepseek_v4_mhc.py` 在本分支的
改动全部是 A3 支持（见 §7）。

| 修改点 | 位置 | 内容 |
|---|---|---|
| 跨层入口 | `models/deepseek_v4.py:4158-4171` | `_forward_layers_hc_pre_from_prev(..., prev_pre=None)`；删除原 `assert pp_group.world_size == 1` |
| 初始化 | `models/deepseek_v4.py:4245` | `state = mhc.HcState(hidden_states, pre=prev_pre)` |
| 非 first rank 接收 | `models/deepseek_v4.py:4645-4663` | 从 `pp_proxy_tensors.tensors["mhc_pre"]` 取；`pp_size>1` 且缺失、dtype 非 fp32、shape 非 `(T, hc_mult)` 均抛错 |
| 非 last rank 发送 | `models/deepseek_v4.py:4803-4817` | 返回 `{"hidden_states": flatten(1), "mhc_pre": last_pre}`；断言 `last_pre` 存在且 fp32 |
| CUDA 关闭跨层融合 | `models/deepseek_v4.py:4414-4419`，`models/deepseek_v4_boundary.py:104` | 非 last rank 的最后一层传 `allow_cross_layer_fusion=False`；`HcUpdate.post` 据此把 `next=None`，先 materialize 本 stage 的 post 再 relay |

不变量：

- `mhc_pre` 与 `hidden_states` 行数一致（`[T, hc_mult]` vs `[T, hc_mult*H]`），
  dtype 固定 fp32，跨 IPC 不做精度转换。
- 只有 `hc_pre_from_prev_sublayer=True` 的模型才传 `mhc_pre`；
  layer 不跨 stage 时该分支不生效。
- NPU 侧无需新逻辑：`models/deepseek_v4_platform_stage.py:133`
  `NonCudaDecoderMixin.forward_hc_pre_from_prev` 直接消费并传递 `state.pre`。

### 2.2 PP proxy 缓冲区 `mhc_pre`【PP 专属】

| 修改点 | 位置 | 内容 |
|---|---|---|
| 宽度解析 | `model_executor/model_runner_components/misc_utils.py:104` | `pp_size>1 and pp_rank>0 and hc_pre_from_prev_sublayer -> hc_mult`，否则 0（first rank 不接收） |
| ModelRunner 委托 | `model_executor/model_runner.py:926` | `get_pp_proxy_hc_pre_width()` |
| eager 分配 | `model_executor/runner/base_runner.py:71-163` | `pp_proxy_tensors["mhc_pre"] = zeros((max_num_token, width), fp32)` |
| eager 透传 | `model_executor/runner/base_runner.py:498` | `pp_proxy_mhc_pre_width=mr.get_pp_proxy_hc_pre_width()` |
| decode graph | `model_executor/runner/decode_cuda_graph_runner.py:582` | 传入宽度 |
| prefill graph | `model_executor/runner/prefill_cuda_graph_runner.py:514` | 传入宽度 |
| dummy proxy | `managers/scheduler_components/dynamic_chunk_sizer.py:217` | dynamic chunk profiling 的 dummy proxy 同步加宽 |
| 分配函数 | `model_executor/runner_utils/buffers.py:67-109` | `allocate_pp_proxy_tensors(..., pp_proxy_mhc_pre_width)`；`mhc_pre` 用 `max_hidden_tokens`，与 `hidden_states` 同界 |

不变量：

- 所有 rank 的 proxy key 集合与形状必须一致；`mhc_pre` 只在 `pp_rank>0` 存在。
- `mhc_pre` 与 `hidden_states` 共用 `max_hidden_tokens`（含 CP 切分后的上界），
  否则 relay 端形状不一致。
- graph replay 的按 key 拷贝会原地刷新 `mhc_pre`；前提是 `prepare_replay`
  不走跳过 `load_batch` 的快路径（见 4.1）。

### 2.3 切分校验【PP 专属】

模型约束（约束来源见 README §2）：

- ratio 1/2 层读取最近的同 ratio `kv_source_layer_ids` 层；跨 stage 后
  KV 池构造/访问失败。
- 低 ratio 层的 top-k 由最近的同 ratio `index_source_layer_ids` 层发布，
  跨 stage 后读不到；两组 source 不必重合。
- `candidate_source_layer_id` 发布的 candidate blocks 被所有后续层消费。
- `dspark_target_layer_ids` 必须在 last stage（draft 只在那里构造）。
- checkpoint 带 vision tower 时 PP 必须配 `--language-model-only`。

| 修改点 | 位置 | 内容 |
|---|---|---|
| 切分算术抽取 | `distributed/utils.py:110-176` | 抽出 `parse_pp_layer_partition`、`compute_pp_indices`；`get_pp_indices` 委托，校验与运行时共用同一套算术；`pp_size==1` 仍忽略 partition |
| 环境变量注册 | `environ.py:782` | `SGLANG_PP_LAYER_PARTITION` 迁入 `envs`；`get()` 每次读 `os.environ`，测试可 patch |
| 每 rank 范围 | `arg_groups/deepseek_v4_hook.py:321-336` | `compute_pp_ranges`，规则与运行时一致 |
| ratio 归一 | `:339-344` | `_model_layer_ratios` 截掉 `compress_ratios` 末尾 nextn 项 |
| source 禁区 | `:357-381` | `_block_source_consumer_groups`：阻塞 `(source, 最后一个同 ratio 消费者]`，允许中间夹其它 ratio |
| 合法边界 | `:384-416` | `legal_split_points`：kv source + index source + candidate + DSpark capture（+ vision 单独校验） |
| 建议切分 | `:419-454` | `suggested_partition`：合法边界上的有界 DP，最小化 `(最大 stage 层数, 层数平方和)` |
| 错误信息 | `:457-485` | 非法 partition、具体约束、合法边界列表、`--pp-size` 下的建议 |
| 主校验 | `:488-575` | `validate_dsv41_pp_partition`：vision、每 stage 的 source 可达性、candidate 与消费者同 stage、DSpark capture 在 last stage |
| 参数期挂载 | `:590-601` | `validate_deepseek_v41_features` 调用主校验 |
| 解除全局拒绝 | `:652-665`（unsupported 列表） | 移除 `("pipeline parallelism", cfg.pp_size > 1)` |
| bounded replay 互斥 | `:736` | PP 移入 `enable_decoder_swa_bounded_replay` 的 incompatible 列表（自动开启谓词的 `not any(...)` 已覆盖，无需重复判断） |
| 运行期二次校验 | `model_executor/model_runner_components/spec_aux_hidden_state.py:202-228`，调用点 `:320` | `validate_pp_capture_layers`：按 draft 配置解析出的实际捕获层与真实 partition 校验，覆盖内置/外置 draft |
| DSpark PD 后端白名单 | `arg_groups/deepseek_v4_hook.py:684-692` | PD 投机校验允许 NPU 的 `ascend` 传输后端 |

### 2.4 last stage 构建 embedding【PP + spec】

| 修改点 | 位置 | 内容 |
|---|---|---|
| embedding 条件 | `models/deepseek_v4.py:3942` | `pp_group.is_first_rank` 改为 `deepseek_v2.pp_stage_needs_embedding(pp_group, speculative_algorithm)`（first rank，或 spec 下的 last rank） |
| 权重加载跳过 | `models/deepseek_v4.py:5851` | `.embed_tokens.` 的跳过条件同步改为该判定 |
| 权重检查跳过 | `models/deepseek_v4.py:5995-5998` | `skipped_checking_patterns` 同步 |

原因：DSpark 的 draft 只在 last stage，且与 target 共享 embed/head
（README §2.3）。`pp_stage_needs_embedding` 本身在 `models/deepseek_v2.py:3041`，
为上游已有函数。

## 3. NPU attention backend【stage 化】

文件：`hardware_backend/npu/attention/ascend_dsv4_backend.py`。

| 修改点 | 位置 | 内容 |
|---|---|---|
| stage ratios | `:2922-2959` | `stage_ratios` 从 `token_to_kv_pool.stage_layer_ids` 派生；`_dsv4_stage_layer_ids`、`_dsv4_unique_compress_ratios`、`_dsv4_has_c4`、`_dsv4_has_c128` 只反映本 stage（`_dsv41_low_ratios()` 因此自动收窄；该函数本身上游已有，分支只加注释） |
| 全局信号保留 | `:2943` | `_dsv4_has_low_ratio = 模型为 V4.1 且非 draft worker`；它同时是"走 ds41 算子族"的信号，ratio-0 层也依赖它 |
| candidate 消费者 | `:2339-2348` | `_dsv41_has_candidate_consumers()` 增加 `layer in _dsv4_stage_layer_ids` 过滤；切分校验保证 candidate 与消费者同 stage |
| graph 缓冲分配 | `:3332-3410` | 非 V4.1 的每个 V4 stage 都分配 SWA 的 `c1a` dispatch table（含全 ratio-0 stage）；V4.1 stage 由上游已有的 `_init_dsv41_graph_metadata` 按本 stage ratios 建表；`c4`/`c128` 的 page table、loc、padding、metadata、`c4_topk_indices` 分别随 `_dsv4_has_c4`/`_dsv4_has_c128` 分配 |
| 刷新路径 | `:3951`、`:4062-4128` | verify 刷新与 `_clear_v4_verify_metadata` 只遍历本 stage 存在的 c4/c128；cycle state 表的条件加入 `_dsv4_has_low_ratio` |
| verify 模式 | `:3018-3044` | DSV4.1 只接受 STATIC 与 CAP_ACCEPT（共用 STATIC 布局）；COMPACT 启动拒绝；`_ensure_dsv41_static_verify_mode` 为本分支新增 |

下游消费者无需改动：compressor / indexer / loc / top-k 的循环上游本就遍历
`_dsv41_low_ratios()`（如 `:1891`、`:2101`），输入集合收窄后自动按 stage 生效，
这些循环不是本分支的新增逻辑。

注意：A5 的 64 query-head 断言（`:2944-2951`）是 A5 算子约束，不是 PP 逻辑。

## 4. NPU graph 与产物

### 4.1 replay 路径【PP 专属】

`hardware_backend/npu/graph_runner/npu_cudagraph_backend.py:218`：
`runner.buffers.pp_proxy_tensors is not None` 时 `prepare_replay` 直接返回
False，强制走通用 `load_batch`。原因：pre-planned 快路径只刷新
input_ids/positions，不执行 `load_batch` 的 PP 准备，会漏刷
`hidden_states` / `mhc_pre`。`pp_size=1` 保留原快路径。

### 4.2 HCCP warm-up【非 PP，graph 稳定性】

`npu_cudagraph_backend.py:70-79`：DSV4 + TP>1 时在第一个自定义 AICPU
metadata kernel 之前做一次 all-reduce。未初始化 HCCP 就启动自定义 AICPU
dispatch 会在部分 CANN/驱动栈上失败（507018）。与 PP 无关，但 PP 首次
开图同样受益。

## 5. KV 池与 PD

### 5.1 KV 池【stage 化 + 配套】

| 修改点 | 位置 | 内容 |
|---|---|---|
| stage 切片基座 | `mem_cache/deepseek_v4_memory_pool.py:105-126`、`:1162-1174`、`:2027` | `collect_sources_by_ratio(..., stage=range(start,end))`；`stage_layer_ids`；`source_layer_of` 在本 stage 找不到 source 直接 assert（由 2.3 的校验保证不会发生）。**上游已有** |
| 请求态 layer id | `mem_cache/deepseek_v4_memory_pool.py:1728` | `get_request_state_layer_ids()`：`request_scoped` state pool 的全局层号，与 `get_request_state_buf_infos` 1:1 |
| 连续缓冲 layer id | `hardware_backend/npu/dsv4/dsv4_memory_pool.py:736` | `get_contiguous_buf_layer_ids()`：按 wire 顺序（C4 → ratio 组 → KV/index-K/index-scale）给出与 `get_contiguous_buf_infos()` 1:1 的层号；镜像"空 buffer 跳过"规则 |
| SWA state id | `dsv4_memory_pool.py:789` | `get_state_layer_ids()`：只含 SWA 行 |
| C4 state id | `dsv4_memory_pool.py:799` | `get_c4_state_layer_ids()`：C4 state 组件独立发布，避免"4 个 id 对 2 个指针" |
| A3 BF16 布局 | `hardware_backend/npu/dsv4/dsv4_kv_layout.py`（`V41_BF16`）、`dsv4_memory_pool.py:110`（`_write_dsv41_bf16_rows`） | A3 的 SWA/压缩 KV/index K 使用 BF16 行；A5 保持 `V41_SWA_FP8` / `V41_CMP_FP4`。**配套（非 PP）** |

### 5.2 PD 布局自描述与配对【PP 专属】

| 修改点 | 位置 | 内容 |
|---|---|---|
| pool 层号入口 | `disaggregation/utils.py:1486-1516` | `build_kv_layer_ids` 优先取 `token_to_kv_pool.get_contiguous_buf_layer_ids()`，owner region 无 id 时回退 |
| 配对规则（上游） | `disaggregation/utils.py:1439` | `build_transfer_entry_pairs` 上游已支持 `allow_positional_fallback`，调用点传 `pp_size == 1`；本分支只提供 pool 侧 layer id（上一行）让按 id 配对成立 |
| Mooncake 空 state | `disaggregation/mooncake/conn.py:1179` | 本 rank 的空占位 state 不参与布局比对 |
| 空 request state | `disaggregation/prefill.py:1881-1885` | 本 stage 无 `request_scoped` pool 时 `_request_state_payload()` 返回 None |
| 空占位注册 | `mem_cache/static_kv_storage.py:902-914` | `_model_has_request_state_outside_stage`：全 ratio-0/1 的 stage 仍发布空 `DSV4_REQUEST_STATE`，保持 wire order 位置稳定 |
| state 层号 | `static_kv_storage.py:1022`、`:1075-1087`、`:1108`、`:1251`、`:1332` | SWA / DSV4_REQUEST_STATE / C4 / draft SWA 各自发布自己的 layer id |
| 低 ratio 位置切片 | `disaggregation/utils.py`（`setup_state_kv_args` 发布 `mla_source_layer_ids`）；`disaggregation/common/conn.py` `_slice_low_ratio_dst_for_pp` | 按 ratio 1/2 的 source 组切目的端：`[ratio][kind][source]`，draft 后缀保持在 target 之后；kind 数（A3=2/A5=3）由源端条目数推导。Ascend 上 V4.1 因非 hybrid 自动落入 upstream 的 legacy 位置路径（Ascend 文件无改动） |

### 5.3 握手字段【PP 专属】

`disaggregation/utils.py:1981` `get_dsv41_spec_layout`：`pp_size>1` 时只返回
rank 一致的字段 `{num_draft_tokens, compression_ratios}`，并要求本 rank
至少有一个 target SWA state；`disaggregation/common/conn.py:1557` 的
`layout_keys` 只比较这两个字段。PP=1 仍比较完整字典。

原因：draft runner 与 draft SWA 只在 last stage，per-rank 的层/state 字段
天然不同，不能参与跨 rank 比较。

## 6. 容量规划【PP 专属】

文件：`model_executor/pool_configurator.py`。

| 修改点 | 位置 | 内容 |
|---|---|---|
| 无本地上限哨兵 | `:75` | `_NO_LOCAL_TOKEN_LIMIT = 1 << 62`，PP MIN 规约会忽略它 |
| 全 ratio-0 stage | `:1957-1985` | `bytes_per_full_token == 0` 时先检查固定池（SWA cap / state / request window）是否超 `available_bytes`；超预算报错，否则返回哨兵，由其他 stage 决定流水线容量 |
| 请求数估算 | `:1248`、`:1852-1862` | 无 token 项时取 `_MIN_ESTIMATED_REQUESTS`，避免除零；估算范围常量化为 `_MIN/_MAX_ESTIMATED_REQUESTS` |

## 7. 配套（非 PP）改动

这些与 PP 改动在同一 diff 中，但不是 PP 逻辑，评审/定位时应分开：

| 领域 | 文件 | 内容 |
|---|---|---|
| A3 mHC | `models/deepseek_v4_mhc.py:386`、`:580`、`:670`、`:688`；`hardware_backend/npu/dsv4/dsv4_mhc.py` | `mix_stats_torch` / `post_torch` / `combine_premix`；A3 无 A5 融合算子 |
| A3 attention | `models/deepseek_v4_dspark.py:85` 起；`ascend_dsv4_backend.py` 的 A3 op 族选择 | A3 head 宽度、无 pad、无 q_head_norm 等 |
| A3 量化/算子 | `layers/quantization/modelslim/*`、`layers/moe/moe_runner/ascend.py`、`hardware_backend/npu/moe/activation.py`、`layers/moe/topk.py`、`dsv4/dsv4_rope.py` | W4A8 clip、未量化 MTP experts、gating top-k、RoPE |
| Engram | `layers/engram.py`、`models/deepseek_v4.py:5372` | ModelSlim QuaRot gate basis、图捕获兼容 |
| DSpark draft dispatch | `speculative/dspark_components/dspark_worker_v2.py:255-265`、`:642-656` | 未量化 draft 的 BF16 dispatch 作用域（`SGLANG_DEEPEP_BF16_DISPATCH` / `DEEP_NORMAL_MODE_USE_INT8_QUANT`） |
| 图捕获稳定性 | `hardware_backend/npu/utils.py:248-280` | Dynamo seed trace 规则补丁 |
| A3 容量定价 | `pool_configurator.py` 的 `_npu_low_ratio_bf16` / `_npu_dspark_draft_layers` | A3 BF16 行与 draft SWA 的字节定价 |

## 8. 文件清单（PP 相关）

| 文件 | 改动摘要 | 分类 |
|---|---|---|
| `models/deepseek_v4.py` | mHC `pre` 交接、stage 边界关融合、last stage embed | PP 专属 |
| `models/deepseek_v4_mhc.py` | A3 mHC torch 实现（`mix_stats_torch` / `post_torch` 入口）与 `combine_premix`；`HcState.pre` 为上游已有 | 配套 |
| `distributed/utils.py` | `parse_pp_layer_partition` / `compute_pp_indices` | PP 专属 |
| `arg_groups/deepseek_v4_hook.py` | 切分校验、解除 PP 拒绝、bounded replay 互斥、PD 后端白名单 | PP 专属 |
| `environ.py` | `SGLANG_PP_LAYER_PARTITION` 注册 | PP 专属 |
| `model_executor/model_runner_components/misc_utils.py` | `resolve_pp_proxy_hc_pre_width` | PP 专属 |
| `model_executor/model_runner.py` | `get_pp_proxy_hc_pre_width` 委托 | PP 专属 |
| `model_executor/runner_utils/buffers.py` | `mhc_pre` 缓冲分配 | PP 专属 |
| `model_executor/runner/base_runner.py` | eager proxy 分配 | PP 专属 |
| `model_executor/runner/decode_cuda_graph_runner.py` | decode graph 宽度透传 | PP 专属 |
| `model_executor/runner/prefill_cuda_graph_runner.py` | prefill graph 宽度透传 | PP 专属 |
| `managers/scheduler_components/dynamic_chunk_sizer.py` | dummy proxy 宽度 | PP 专属 |
| `model_executor/model_runner_components/spec_aux_hidden_state.py` | `validate_pp_capture_layers` | PP 专属 |
| `hardware_backend/npu/attention/ascend_dsv4_backend.py` | metadata / 图缓冲 / candidate 按 stage 收窄（另含 A3 op 族） | stage 化 + 配套 |
| `hardware_backend/npu/graph_runner/npu_cudagraph_backend.py` | PP 下回退 `load_batch`；HCCP warm-up | PP 专属 + 非 PP |
| `hardware_backend/npu/dsv4/dsv4_memory_pool.py` | PD layer id 发布；A3 BF16 行写入 | PP 专属 + 配套 |
| `hardware_backend/npu/dsv4/dsv4_kv_layout.py` | `V41_BF16` 布局 | 配套 |
| `mem_cache/deepseek_v4_memory_pool.py` | `get_request_state_layer_ids` | PP 专属 |
| `mem_cache/static_kv_storage.py` | state layer id、空占位 | PP 专属 |
| `disaggregation/utils.py` | `build_kv_layer_ids` 接入 pool layer id、`mla_source_layer_ids` 发布、rank-uniform 握手 | PP 专属 |
| `disaggregation/common/conn.py` | PP 下的 `layout_keys`、低 ratio 位置切片 | PP 专属 |
| `disaggregation/mooncake/conn.py` | 空 state 跳过 | PP 专属 |
| `disaggregation/prefill.py` | 空 request state payload | PP 专属 |
| `model_executor/pool_configurator.py` | 无本地上限哨兵、固定池检查（另含 A3 定价） | PP 专属 + 配套 |
| `test/registered/unit/npu/test_dsv41_pp_partition.py` | 切分校验与 PD 配对单测 | PP 专属 |
| `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py`、`test_npu_dsv41_low_ratio_backend.py`、`test_dsv41_dspark_metadata_rebase.py`、`graph_runner/test_npu_cudagraph_backend.py` | stage ratio、图缓冲、graph 回退 | PP 相关 |

其余 A3/DSpark 文件（`models/deepseek_v4_dspark.py`、`layers/engram.py`、
`layers/moe/*`、`layers/quantization/modelslim/*`、`hardware_backend/npu/utils.py`、
`hardware_backend/npu/moe/*`、`hardware_backend/npu/dsv4/dsv4_rope.py`、
`speculative/dspark_components/dspark_worker_v2.py` 等）属于第 7 节的配套改动，
完整列表见 `git diff --stat upstream/main..dsv41_pp`。

## 9. 上游基座（本分支未改，勿混淆）

PP 依赖以下上游实现；问题定位时不要把它们的现有行为算作本分支修改：

| 机制 | 位置 |
|---|---|
| PP proxy 张量容器 | `model_executor/forward_batch_info.py:2391` `PPProxyTensors` |
| PP 调度循环 / proxy 收发 | `managers/scheduler_pp_mixin.py`（`_pp_recv_proxy_tensors` / `_pp_send_proxy_to_next_stage`） |
| NPU pp2 合并收发、pp>2 parity 顺序 | `managers/scheduler_pp_mixin.py`、`distributed/parallel_state.py:1934` |
| 层列表 stage 切片 | `layers/utils/layer_list.py` `make_pp_layers` |
| lagged mHC `HcState.pre` 语义与消费 | `models/deepseek_v4_mhc.py:196-250`（`run_attn_post` / `run_moe_post` 产出下一子层的 pre-mix）；`models/deepseek_v4.py` 上游已有 `_forward_layers_hc_pre_from_prev` 返回 `state.pre` 与 last rank 的 `mhc_combine` 收尾 |
| NPU 非 CUDA 层的 `forward_hc_pre_from_prev` | `models/deepseek_v4_platform_stage.py:133` |
| KV 传输按 layer id 配对 | `disaggregation/utils.py:1439` `build_transfer_entry_pairs`（`allow_positional_fallback`） |
| `pp_stage_needs_embedding` | `models/deepseek_v2.py:3041` |
| DSpark `_hosts_draft` / idle proxy 透传 | `speculative/dspark_components/dspark_worker_v2.py:206`、`:857-867` |
| DSpark + PP 能力边界 | `speculative/speculative_hook.py`、`arg_groups/validation_hook.py` |

## 10. 测试与验收

- `test/registered/unit/npu/test_dsv41_pp_partition.py`（16 项）：
  合法边界、文档中的 PP2/3/4/6 切分、默认 PP4 被拒、kv/index source 分离、
  candidate、DSpark capture、缺失 `dspark_target_layer_ids`、运行期 capture
  校验、PD 按 layer id 配对 / 无 id 拒绝。
- stage 化相关：`test_npu_ascend_dsv4_backend.py`（68 项，含 SWA/C4 id 1:1）、
  `test_npu_dsv41_low_ratio_backend.py`（102 项）、
  `test_dsv41_dspark_metadata_rebase.py`（19 项，含全 ratio-0 / 仅 C4 /
  仅 C128 stage 的图缓冲）、`test_npu_cudagraph_backend.py`（6 项）。
- `test_pp_hybrid_kv_transfer.py` 新增 8 项：DSV4.1 ratio 1/2 位置切片的
  PP2/PP4、draft 后缀、无 source stage、A3/A5 布局不一致拒绝；以及 Ascend
  路由 3 项（非 hybrid 走位置路径、PP2 两段、draft 后缀）。
- 实机验收顺序与命令见 `README.md` §5；尚未执行的项目见 `README.md` §7。

## 11. 能力边界

支持/拒绝的组合见 `design.md` §1 与 `README.md` §3。本文只补充：
PP 下 **DSpark 仅支持 PD prefill 形态**，非 PD 的 PP+投机只支持 EAGLE 的
PP spec relay；这两点由上游校验保证，本分支没有放宽。
