# DeepSeek-V4.1 NPU 平台 PP 适配：分析与设计

> 代码基线：`/home/w00889861/dsv4/sglang`，分支 `dsv41_pp`，已 rebase 到
> `upstream/main` `e183c415bd`（2026-11-20）。上游同日合入 DSV4.1 A5 支持
> （#47660），因此分支的 A5 系列以上游版本为准，分支只在其上维护 A3、
> DSpark 与 PP 改动；公共 mHC `pre` 交接与切分校验仍由本分支提供。
> 本文给出分析与总体设计；实现细节、不变量、失败模式与验收方式见同目录
> [`design.md`](design.md)。

## 0. 实现状态（2026-11-20，rebase 后）

- 普通 PP（无投机）与 PD 的 P 端 PP + DSpark 已实现；公共代码只增加 mHC `pre`
  交接、proxy 缓冲与切分校验，其余适配在 NPU 内部。
- 非 PD DSpark + PP、vision + PP、PP + decoder bounded replay、PP + overlap
  schedule 均由配置校验拒绝。
- rebase 后 CPU 单测全量通过（PP 切分、NPU attention、graph、A3 DSpark、
  Engram），实机 e2e 待验（§5.0）。逐条设计、不变量、评审结论见
  [`design.md`](design.md)。

## 1. 结论先行

### 1.1 DSV4.1 在 GPU 上原本是否支持 PP？

**原始实现不支持。** 有两道阻断（本分支已解除）：

1. 配置层：`python/sglang/srt/arg_groups/deepseek_v4_hook.py` 的
   `validate_deepseek_v41_features` 把 `pp_size > 1` 列为 unsupported，直接抛
   `DeepSeek-V4.1 does not support pipeline parallelism yet`。
2. 模型层：`python/sglang/srt/models/deepseek_v4.py` 的
   `_forward_layers_hc_pre_from_prev` 有
   `assert self.pp_group.world_size == 1, "pre-mix hand-off across PP is not wired"`。
   DSV4.1 的 `hc_pre_from_prev_sublayer` 恒为 True，因此必然走该函数。

根因不是"层切分"本身，而是 DSV4.1 的 lagged mHC 需要跨 stage 传递一个额外状态
`HcState.pre`：每个子层的 pre-mix 系数由上一个子层产生、被下一个子层消费。PP 只传
`hidden_states`，跨 stage 的 `pre` 没有接线。

本分支在公共代码中补上了该交接与对应的切分校验（§3.1）。NPU 已完成实现与单元
验证，GPU 侧的同一套公共路径尚未实机验收（§7）。

### 1.2 历史 PR #39820 的作用

`https://github.com/sgl-project/sglang/pull/39820`（`[NPU][DSV4]dsv4 enable cpp`）
已合入 `main`，并且是当前分支的祖先。它**不是** DSV4.1 PP 的主体实现，只修了两处
DSV4（非 4.1）PP 的实际问题：

1. KV 池按 PP stage-local 层号索引（`_swa_local_layer_id`）；
2. prefill PP + decode DSpark 长序列时 sparse attention op 的选择。

这两处在当前分支已经存在（`hardware_backend/npu/dsv4/dsv4_memory_pool.py` 的
`get_key_buffer` / `get_swa_buffer` / `set_swa_buffer`）。DSV4（非 4.1）的 PP
基础设施原本就在 mainline：`make_pp_layers`、`PPProxyTensors`、KV 池 stage 切片、
`SGLANG_PP_LAYER_PARTITION` 等都已具备。

## 2. 模型约束：为什么 DSV4.1 需要专门的 PP 切分校验

其他模型的每层是自包含的，PP 均匀切分或用户任意切分都不会出错，因此没有模型级
PP 校验（只有 `get_pp_indices` 的通用均分）。DSV4.1 不同，它有三类**跨层共享、
按 stage 物化**的状态，另有一条 vision 构造限制。

真实 checkpoint（`/mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8`）的关键字段：

```
num_hidden_layers          = 40
compress_ratios            = [0,0] + [2]*18 + [1]*20 + [0,0,0]   # 末尾 3 项给 nextn
kv_source_layer_ids        = [2, 8, 14, 20]
index_source_layer_ids     = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id   = 20
candidate_topk_blocks      = 2048
candidate_block_size       = 8
hc_mult                    = 4
num_nextn_predict_layers   = 3
dspark_target_layer_ids    = [37, 38, 39]
vision_config.num_hidden_layers = 32
```

### 2.1 压缩 KV 的 source 组

ratio-1/2 的层不自带压缩存储，而是读"最近的、同 ratio 的 kv_source 层"写出的缓存
（`mem_cache/deepseek_v4_memory_pool.py`）：

- `collect_sources_by_ratio(..., stage=range(start_layer, end_layer))` 只收集
  本 stage 内的 source；
- `source_layer_of(layer_id)` 在本 stage 内找不到 source 时直接 assert；
- `layer_mapping` / `compress_state_pools` 只填充 `[stage_start, stage_end)`。

对本 checkpoint：

- ratio 2：source `[2, 8, 14]`，组为 `[2,8)`、`[8,14)`、`[14,20)`；
- ratio 1：source `[20]`，组为 `[20,40)`。

PP 边界若落在组内部（例如 layer 10），stage 1 的 layer 10 需要 stage 0 的 source 8，
KV 池会在构造/访问时失败。

### 2.2 candidate source

`candidate_source_layer_id=20` 发布的 candidate blocks 被所有后续层消费
（`DeepseekV41Indexer.uses_candidates = 0 <= candidate_source < layer_id`）。
NPU 的 candidate blocks 与 QSLI 缓存是 per-forward、per-stage 状态
（`ascend_dsv4_backend.py` 的 `fm.dsv41_candidate_blocks`、`_dsv41_qsli_cache`），
跨 rank 的消费者读不到。因此 layer 20 与 21..39 必须同 stage——这正好与 ratio-1
组 `[20,40)` 重合。

### 2.3 DSpark 捕获层

DSpark 的 draft 只在最后一个 PP rank 构造
（`DSparkWorkerV2._hosts_draft = pp_group.is_last_rank`）。
`set_dspark_layers_to_capture` 在非 last rank 直接 return；若
`dspark_target_layer_ids` 中有层不在 last stage，draft 会静默缺少 aux hidden，
精度下降但不会断言。因此 PP 校验必须要求 `[37,38,39]` 全部落在 last stage。

### 2.4 视觉塔

该 checkpoint 的 `vision_config.num_hidden_layers=32`。模型构造处明确拒绝
"vision + PP"（`DeepseekV4ForCausalLM.__init__`），因此 PP 必须配合
`--language-model-only`（PD 场景该 flag 已被支持）。

### 2.5 合法切分边界

由上述约束，40 层的合法 stage 边界为：

```
{1, 2, 8, 14, 20, 40}
```

- `1` 合法：layer 0/1 是 ratio 0，无 source 依赖；
- `20..39` 必须整段同 stage（ratio-1 + candidate source 组）；
- 除 kv_source 外，`index_source_layer_ids`（产出低 ratio 层注意力所用 top-k 的层）
  也必须与其后的同 ratio 层同 stage。该 checkpoint 的 index source
  `[2,8,14,20,24,28,32,36]` 与 kv source 在 2/8/14/20 重合，因此合法边界不变；
- PP=2 默认均分 20/20 合法；
- PP=4 默认 10/10/10/10 非法（边界 10 切开 source 8 的组，30 切开 20..39 组）。

可用 `SGLANG_PP_LAYER_PARTITION` 显式指定，例如：

| PP | 建议 partition |
|----|----------------|
| 2  | `20,20`（默认即可） |
| 3  | `14,6,20` 或 `8,12,20` |
| 4  | `8,6,6,20`（或 `2,6,6,26`） |
| 5  | `2,6,6,6,20` |
| 6  | `1,1,6,6,6,20` |

负载上 20..39 必须同 stage，PP 越大最后一段越重，这是模型结构决定的。

## 3. 设计总览

目标部署（用户确认）：

1. **阶段 1**：PD 分离的 P 节点同时开启 PP + DSpark（现有校验允许的形态：
   `disaggregation_mode=prefill`，`SGLANG_ENABLE_PP_SPEC=0`）。
2. **阶段 2（不做）**：单机非 PD 同时开启 PP + DSpark。上游/GPU 明确拒绝该
   组合：`speculative_hook.py` 要求 DSpark+PP 只能是 PD prefill，
   `validation_hook.py` 禁止 DSPARK 配 `SGLANG_ENABLE_PP_SPEC`。本分支保持
   同样的能力边界；非 PD 的 PP+投机只支持 EAGLE（`SGLANG_ENABLE_PP_SPEC=1`）。
3. 支持多 PP size 与 `SGLANG_PP_LAYER_PARTITION`。
4. A3 先开发验证，再 A5；最少 `is_npu` 分支，最大复用 GPU/公共代码。

设计分成三层：

### 3.1 公共代码：补上 DSV4.1 PP 的两个缺口

**(a) mHC `pre` 跨 stage 交接**

`HcState.pre` 是 fp32 `[T, hc_mult]`，由上一个子层的 post 产生、被下一个子层的
combine 消费。PP 边界只需传它（`input`/`stats` 是层内的，不跨层）。

- `DeepseekV4Model._forward_impl` 在非 first rank 从
  `pp_proxy_tensors.tensors["mhc_pre"]` 取出上一 stage 的 pre，传给
  `_forward_layers_hc_pre_from_prev(prev_pre=...)`；
- 该函数用 `mhc.HcState(hidden_states, pre=prev_pre)` 初始化；
- 非 last rank 返回 `PPProxyTensors({"hidden_states": ..., "mhc_pre": last_pre})`；
- last rank 仍用 `last_pre` 做 `mhc_combine`（现有逻辑不变）；
- CUDA 路径在 stage 边界最后一层关闭跨层融合
  （`allow_cross_layer_fusion=False`），避免尝试融合下一 rank 的层；
- NPU 的 `NonCudaDecoderMixin.forward_hc_pre_from_prev` 直接消费 `state.pre`，
  无需新逻辑。

**(b) PP proxy 缓冲区**

沿用现有的 `pp_proxy_topk_size` / `pp_proxy_residual_num_blocks` 扩展模式：

- `misc_utils.resolve_pp_proxy_hc_pre_width`：仅 `pp_size>1`、`pp_rank>0`、
  `hc_pre_from_prev_sublayer=True` 时返回 `hc_mult`；
- `ModelRunner.get_pp_proxy_hc_pre_width` 薄委托；
- `allocate_pp_proxy_tensors` 增加 `mhc_pre`（fp32，token-major）；
- decode/prefill graph runner 与 eager buffer 透传；
- graph replay 的按 key 拷贝逻辑天然支持，无需新图逻辑。

**(c) 配置校验**

`validate_dsv41_pp_partition`（内联在 `deepseek_v4_hook.py`）：

- 纯配置计算，无 torch/设备依赖；
- 按 `get_pp_indices` 相同规则计算每个 rank 的 `[start,end)`；
- 校验 kv_source / index_source 同 stage、candidate source 同 stage、DSpark 捕获层在 last
  stage（建议切分同样纳入该约束）、vision 需 `--language-model-only`；
- kv_source 的约束按“source 到最后一个同 ratio 消费者”计算，允许中间夹其它
  ratio 的层；
- 错误信息给出合法边界与建议 partition，建议结果由同一约束集导出。

**为什么不新增独立文件**：其他模型没有这类按 stage 物化的跨层共享状态，所以没有
模型级 PP 校验文件；DSV4.1 需要的是"校验逻辑"而非"独立模块"。为保持最小新增
文件，校验函数内联在 DSV4 专用 hook 中。

### 3.2 NPU 适配（不改公共语义）

**(a) attention backend 按 stage 收窄 ratio**

`DeepseekV4AscendAttnBackend` 原先用全局 `hf.compress_ratios` 构建 metadata 与
graph buffer。PP 下某个 stage 可能完全没有 ratio 4/128/1/2，访问空池会出错。

修改方式（不新增 `is_npu` 分支）：

- 从 KV 池的 `stage_layer_ids` 派生本 stage 实际存在的 ratios，写入
  `_dsv4_unique_compress_ratios`；
- metadata / graph buffer / candidate 消费者判断使用 stage ratios 或 stage 层集合；
- `_dsv4_has_low_ratio` 保持**模型全局**：它不仅是"是否有 low ratio 层"，还是
  "是否走 DSV4.1 ds41 算子族"的信号，ratio-0 层也依赖它。

**(b) NPU graph 快路径**

`NPUCudaGraphBackend.prepare_replay` 的 pre-planned 路径不刷新
`pp_proxy_tensors`。PP 下直接返回 False，走通用 `load_batch` 路径，确保
`hidden_states` / `mhc_pre` 等 proxy 输入在图 replay 前被刷新。

### 3.3 阶段 2：单机非 PD PP + DSpark（不做，已移除）

capability 与 upstream/GPU 对齐后，这一形态不再实现：DSpark + PP 只允许 PD
prefill（`disaggregation_mode=prefill`），且禁止 `SGLANG_ENABLE_PP_SPEC`。

本分支曾实现过 tail-draft/relay 版本（`_PPTailState`、prefill/decode tail
draft、`prepare_pp_verify_input`、`_pp_dspark_rebuild_verify_input`、relay 的
KV 预留等）；为与 GPU 能力一致，这些代码已整体回退，仅保留上游同款的 PD
prefill 形态（draft 在最后一个 PP stage，其他 stage 只跑 target，无 relay）。

2026-11-19 评审再次确认该形态是合入阻断：末级 verify 未透传 proxy，且
`_pp_tail_state` 只有一个 worker 级槽位，多 microbatch 交错时会互相覆盖。
当前实现保持“仅 PD prefill”，评审结论与修复见 [`design.md`](design.md) §11。

## 4. 实现文件清单

rebase 后，分支相对 `upstream/main` 的净改动（含测试与本文档）可用
`git diff --stat upstream/main..dsv41_pp` 查看。逐条修改点、动机与精确到函数的
代码位置见 [`modifications.md`](modifications.md)；本节只给按职责分组的文件清单：

| 组 | 文件 | 改动 |
|----|------|------|
| 公共 mHC | `models/deepseek_v4.py`、`models/deepseek_v4_mhc.py` | `HcState.pre` 跨 stage 交接；stage 边界关闭跨层融合；last rank 的 `lm_head` / 共享 embed；A5 `mhc_pre/post_dpskv4` 包装 |
| PP proxy | `model_runner_components/misc_utils.py`、`model_runner.py`、`runner_utils/buffers.py`、`runner/base_runner.py`、`runner/decode_cuda_graph_runner.py`、`runner/prefill_cuda_graph_runner.py`、`scheduler_components/dynamic_chunk_sizer.py` | `mhc_pre` 宽度解析与 fp32 缓冲分配，eager / graph / dummy 三条路径共用 |
| 切分校验 | `arg_groups/deepseek_v4_hook.py`、`distributed/utils.py` | 抽出 `compute_pp_indices` / `parse_pp_layer_partition` 供校验与运行时共用；`validate_dsv41_pp_partition` 校验 kv/index source 消费区间、candidate、DSpark capture、vision，并给出合法切分建议 |
| NPU attention | `hardware_backend/npu/attention/ascend_dsv4_backend.py`、`hardware_backend/npu/dsv4/dsv4_memory_pool.py`、`dsv4/dsv4_rope.py`、`npu/moe/topk.py`、`npu/utils.py` | metadata / graph buffer / candidate 判断按 stage 收窄；A3/A5 算子族与 DSL 包装；A5 `cann_ops_transformer` 注册 |
| NPU graph | `hardware_backend/npu/graph_runner/npu_cudagraph_backend.py` | PP 下回退通用 `load_batch`，保证 proxy 刷新；HCCP warm-up |
| PD | `disaggregation/{common,mooncake}/conn.py`、`disaggregation/prefill.py`、`disaggregation/utils.py`、`mem_cache/deepseek_v4_memory_pool.py`、`mem_cache/static_kv_storage.py` | KV 布局自描述（全局 layer id / item length）、hybrid 按层 id 配对、DSV4.1 低 ratio 按 source 组位置切片、空占位、rank-uniform 握手、注册期步长校验（双方提供 lens 时） |
| 容量 | `model_executor/pool_configurator.py` | 全 ratio-0 stage 的固定池预算检查与“无本地上限”哨兵，交由 PP MIN 统一规约 |
| 配套修复 | `layers/engram.py`、`layers/moe/moe_runner/ascend.py`、`layers/quantization/modelslim/*`、`speculative/dspark_components/dspark_worker_v2.py` | padded verify 行哈希、W4A8 clip、未量化 MTP experts、draft BF16 dispatch、idle proxy 透传 |
| 测试 | `test/registered/unit/npu/test_dsv41_pp_partition.py`、`test_npu_ascend_dsv4_backend.py`、`test_npu_cudagraph_backend.py` | 切分校验、NPU attention、graph |

`speculative_hook.py`、`validation_hook.py`、`scheduler.py`、`scheduler_pp_mixin.py`、
`dspark_verify.py` 最终**没有**相对父提交的改动：DSpark + PP 的能力边界与
PP 调度循环沿用上游现有实现。

## 5. 验证计划

### 5.0 当前验证状态（2026-11-20，rebase 后）

已执行（CPU / NPU 单元级）：

- `test_dsv41_pp_partition.py`：16/16，覆盖合法边界、文档 partition、非法切分、
  vision、DSpark capture、交错 ratio 的 kv/index source 消费区间、运行时捕获层
  与 PD 按 layer id 配对；
- `test_npu_ascend_dsv4_backend.py`：68/68（含 SWA/C4 层 id 1:1）；
- `test_npu_dsv41_low_ratio_backend.py`：102/102；
- `test_dsv41_dspark_metadata_rebase.py`：19/19（含全 ratio-0 / 仅 C4 / 仅 C128
  stage 的图缓冲）；
- `test_npu_cudagraph_backend.py`：6/6；
- A3 / DSpark / Engram / graph runner 等全部 NPU 单测通过；唯一例外
  `test_npu_seed_trace_rules.py` 的 2 项在 rebase 前的分支上以同一 torch
  构建同样失败（Dynamo 规则缓存冲突），非本次改动引入；
- 全部改动文件 `py_compile`、`git diff --check` 通过；
- 行为抽查：DSpark + PP 在非 PD 下被拒绝；`SGLANG_RAGGED_VERIFY_MODE` 的
  `static` / `cap-accept` 通过、`compact` 被拒绝。

尚未执行：实机 e2e（普通 PP、PD PP + DSpark）。需要空闲设备、匹配的
`sgl_kernel_npu` wheel 与 CANN 9.2 运行时；命令见下文。

### 5.1 A3 单机 PP（无 spec，先验证 PP 本身）

16 卡：`--tp-size 8 --pp-size 2`（先不开 DP attention）。

```bash
cd /home/w00889861/dsv4/sglang
PYTHONPATH=$PWD/python:$PYTHONPATH python3 -m sglang.launch_server \
  --model-path /mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8 \
  --device npu --trust-remote-code --attention-backend dsv4 \
  --quantization modelslim --kv-cache-dtype bfloat16 \
  --language-model-only \
  --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"], "quantization_config": null}' \
  --tp-size 8 --pp-size 2 --page-size 128 \
  --mem-fraction-static 0.85 --disable-cuda-graph
```

验证步骤：启动 → curl `/v1/chat/completions` → 长序列（32k/64k）压测 →
50 条 GSM8K（`sgl-eval run gsm8k --num-examples 50`，按模型推荐采样设置）→
与 `--pp-size 1` 基线对比。

多 PP：`--pp-size 4` + `SGLANG_PP_LAYER_PARTITION=8,6,6,20`。

### 5.2 A3 PD：P 节点 PP+DSpark，D 节点 DSpark

P：`--tp-size 8 --pp-size 2 --disaggregation-mode prefill
--speculative-algorithm DSPARK ...`；D：现有 DSpark decode 配置，`pp_size=1`。
沿用 `TestNpuAccuracyMultiNodePdSepTestCaseBase` 模板。本机 16 卡需多节点；
若必须单机，可尝试 P: TP4×PP2=8 卡 + D: TP8=8 卡。

### 5.3 阶段 2（已移除）

不做。非 PD 场景的 PP+投机只支持 EAGLE（`SGLANG_ENABLE_PP_SPEC=1`，
禁止 DP attention）；DSpark+PP 只保留 PD prefill 形态（见 5.2）。

### 5.4 A5

同流程替换 A5 环境，重点验证 `mhc_pre_dpskv4` / `mhc_post_dpskv4` 包装、
`cann_ops_transformer` 注册与 A5 DSpark 共享 embed 路径。

## 6. 已遇到的问题与排查记录

### 6.1 ModelSlim 描述缺失（启动即失败）

现象：模型构造时 `Missing ModelSlim MoE quantization description for layer
model.layers.0.mlp.experts`。

根因：`config.json` 的 `quantization_config={"quant_method":"ascend"}` 让
SGLang 的 `resolve_checkpoint_quant_spec` 命中内联 quant 配置，跳过了
`quant_model_description.json`，于是 `ModelSlimConfig.quant_description` 只有
`quant_method` 等 3 个键。

修法（运行时）：显式覆盖 config 中的 quantization_config：

```
--quantization modelslim
--json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"], "quantization_config": null}'
```

### 6.2 兄弟容器占用显存

本容器内无服务进程，但 16 张卡各被占用约 46–50 GB（`npu-smi` 显示的 PID 在
本容器不可见），模型构造阶段 OOM。e2e 因此阻塞；需等占用释放或换空闲环境。

## 7. 未验证项与风险

1. **非 PD PP+DSpark 不支持**：评审确认末级 verify proxy 未透传、tail 状态只有
   单 microbatch 槽位，当前实现维持“仅 PD prefill”。后续若要支持，需复用
   microbatch 槽位与按 RID 重排，并补齐 verify proxy 透传。
2. **GPU PP 未实机验收**：公共 mHC 交接与切分校验在 CUDA 上同样生效，但本分支
   只在 NPU 验证；合入前需要 GPU PP 冒烟。
3. **数值一致性**：PP 边界使跨层 mHC 融合退化为普通 post，与 pp=1 可能有非
   bitwise 差异（模型本身已声明 batch 组合下非 bitwise 稳定）；用 KL/GSM8K 比较。
4. **NPU pp>2**：仅 pp2 有 `batch_isend_irecv` 快路径，pp>2 走通用 parity 收发，
   需实测多 microbatch 不 hang。
5. **A5 路径**尚未完整验收：需单独验证 `mhc_pre/post_dpskv4`、共享 embed 与
   DSpark draft。
6. **算子版本耦合**：DSL 包装（compressor/indexer、rope、moe top-k、
   `kv_compress_epilog_v2`）要求 CANN 9.2 与匹配的 `sgl_kernel_npu` wheel；
   跨版本部署前先跑导入与首轮 forward。
7. **PD 异构部署**：注册期在双方都提供 per-entry item lengths 时校验步长；
   hybrid MLA 按 layer id 配对，DSV4.1 低 ratio 按 source 组位置配对。
   跨版本/异构部署必须双方同时升级。

## 8. 参考资料

- 详细设计：[`design.md`](design.md)（数据流、不变量、切分算法、PD 布局、
  容量规划、评审结论、测试与验收）。
- 修改点与涉及文件：[`modifications.md`](modifications.md)（逐条修改点、
  动机、函数级位置、文件清单、PP 专属与 A3/DSpark 配套改动的区分）。
- 切分单测：`test/registered/unit/npu/test_dsv41_pp_partition.py`。
- 历史 PR：`https://github.com/sgl-project/sglang/pull/39820`
  （`[NPU][DSV4]dsv4 enable cpp`，已合入 main 且为当前分支祖先）。
- 真实权重：`/mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8`。
- 关键机制：
  - mHC：`python/sglang/srt/models/deepseek_v4_mhc.py`
  - KV 池 stage 切片：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
  - PP 层构建：`python/sglang/srt/layers/utils/layer_list.py`
  - PP relay：`python/sglang/srt/managers/scheduler_pp_mixin.py`、
    `python/sglang/srt/speculative/pp_spec_relay.py`
  - DSpark：`python/sglang/srt/speculative/dspark_components/`
