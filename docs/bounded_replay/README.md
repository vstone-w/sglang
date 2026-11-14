# DSV4.1 Bounded Replay 文档集

本目录固化 DeepSeek-V4.1 两个 "SWA bounded replay" 特性在 Ascend NPU 上的
适配设计。GPU 侧实现是本仓库的基线代码，NPU 侧按"最大复用、改动收敛"原则移植。

## 两个特性（名字相近，语义不同）

| 特性 | 开关 | 默认 | 作用 |
|---|---|---|---|
| Decoder SWA bounded replay | `--enable-decoder-swa-bounded-replay` | GPU 上满足条件时开启 | prefill 时前段层跑完整 extend，`late_layer_start` 之后的层只跑每个请求最后 `SWA_WINDOW=128` 个 token；CUDA 上这 20 层再由行桶图回放 |
| Encoder SWA bounded replay | `--enable-encoder-swa-bounded-replay` | 关闭（实验性） | radix cache 前缀命中时复用 Main KV / Indexer K，但请求私有 SWA window 由重跑最后 ≤128 个缓存 token 重建（fold 进 extend，或单独 replay forward） |

两者可以叠加：decoder replay 只写尾部的 late-layer SWA KV，encoder replay
负责在前缀命中时把这些窗口补回来。

## 术语表

- **SWA_WINDOW**：滑动窗口长度，固定 128（`deepseek_v4_backend.py:175`）。
- **late layer**：`late_layer_start = max(kv_source_layer_ids)` 及其之后的层
  （V4.1-Flash 为第 20~39 层）。这些层只做 SWA + ratio-1 压缩读取，不自行压缩。
- **tail（尾行）**：每个请求 extend 的最后 `min(128, extend_len)` 行；
  **whole-extend** 请求（prompt logprob、非 DSpark 的 FULL hidden states）例外，
  跑整个 extend。
- **tail source KV**：`late_layer_start` 这一层（也是最后一个 kv/index/candidate
  source）先在完整 extend 上写压缩 KV 和 index K，尾段内该层再跳过压缩器。
- **request window（请求窗口）**：encoder replay 的每请求私有 SWA 环形区 +
  固定 workspace；容量 `sliding_window + draft_tokens`，按页对齐。
- **staging / direct ring**：NPU 的两种窗口寻址。prefill/fold 用紧凑
  layout 把历史行与新行 staging 到 workspace；decode/verify 用 per-request
  ring 页表直接读写请求环（`slot*ring_pages + p%ring_pages`），不再 staging。
- **fold**：把命中的最后 ≤128 个缓存 token 拼到 extend 前面，用一次 forward
  完成重建（CUDA `dsv4` 与目标 NPU 都走这条）。
- **request ring / workspace**：窗口的持久环与临时行区；`WindowLayout` 描述
  历史行、新行、floor、commit 掩码。
- **encoder row floor**：fold 行的 SWA 窗口下界 = 该请求的 replay 起点。
- **DSpark aux**：`dspark_layers_to_capture` 指定的目标层注意力输入均值，
  prefill 时按 tail 行捕获，供 draft KV 注入。
- **DP attention**：attention DP（`attn_dp_size > 1`），late-layer 行数需在
  forward 前跨 rank 交换。
- **PD**：prefill/decode 分离；P 角色启用 replay，D 角色继承 KV/窗口状态。
- **A3 / A5**：A3 = arch22（BF16/W8A8），A5 = arch35 / Ascend 950（量化、
  MQSMLA、64 query heads）。

## 数据流总览（decoder replay）

```
Scheduler: enable_decoder_swa_bounded_replay?
   |  (radix: 命中前缀裁掉最后 128 token，保证 late-layer 窗口被重算)
   v
ForwardBatch.init_new
   |  late_layer_num_tokens(rank) 在 forward 前跨 DP rank 交换
   v
AttentionBackend.init_forward_metadata
   |  -- full metadata
   |  -- tail_forward_metadata（仅 tail 行）
   v
model._forward_layers_hc_pre_from_prev
   |-- layers [start, late_layer_start): 全量行
   |-- i == late_layer_start:
   |     write_tail_source_kv(全量行: 压缩 KV + index K)
   |     enter_late_layer_tail -> 切到 tail metadata / tail 行
   |-- late layers: 只跑 tail 行
   |     (CUDA: RowBucketGraphRunner 行桶图; NPU M2: NPUGraph 行桶图)
   |-- exit_late_layer_tail
   v
logits / hidden states 按 tail rows 对齐；DSpark aux 仅 tail 行
```

Encoder replay（E1/E3）在 NPU 上的窗口寻址分两段：

- **prefill / fold**：`NPUWindowLayout` 紧凑布局 staging 到 workspace；
  窗口 floor 由 `encoder_swa_row_floor` 保证；
- **decode / verify**：`NPURequestWindowController.init_decode_direct` /
  `init_verify_direct` 用 per-request ring 页表直接读写请求环
  （`make_ring_page_table`/`RingPageTableCache`，`seqused=seq_lens`），K 直写环、无 workspace
  gather/commit；prefill 仍保留 staging。

## 当前状态（2026-11-19，A3）

- **M1（decoder eager tail）**：A3 已验证；GSM8K 1.0，DSpark accept 3.3–4.7。
- **E1（encoder eager fold）**：A3 已验证；GSM8K 1.0，共享前缀命中
  47.08%；fold 命中路径输出已用 rank 配对探针校验。
- **E3（encoder + DSpark）**：A3 已验证；GSM8K 1.0、命中路径 16/16、
  accept 3.8–4.8；decode/verify direct ring 比 staging TPOT 约 -11%。
- **NPU graph（encoder decode/verify）**：HCCP 初始化修复后已解锁
  （`ensure_hccp_initialized`）；A3 无 replay 与 E3 target verify graph
  均 capture 通过，resolver 不再要求 `--disable-cuda-graph`。DP idle
  replay 已加 commit 保护；E3 graph 的完整精度复跑待做。DSpark draft
  ring 的图捕获已打开，真机验证待补。
- **算子入口**：V4.1 行格式读取走 `sgl_kernel_npu` 的
  `mixed_quant_sparse_flash_mla_dsl` / `quant_lightning_indexer_dsl` /
  `kv_compress_epilog_v2` 等入口（rope 保留旧
  `torch.ops.custom` 的 ImportError 回退）；A3 BF16 路径不受影响。
- **SWA 准入预算**：`DecodePreallocQueue` 按每请求窗口真实占用
  （`swa_per_request_tokens`）封顶，不再按 `max_new_tokens` 全量预留。
  这是公共行为变化，需覆盖原有 GPU/NPU SWA PD 部署的独立验证。
- **M2（decoder 行桶图 / prefill tail graph）**：A3 已验证首版
  （2026-11-17）：NPU 行桶 `NPUGraph`（单请求/rank、128 行）在
  `mem=0.86` + eager decode + DSpark off/on 下均 capture/replay 通过；
  两波 16/16 正确（DSpark aux 同轮验证）。**decode graph 共存已验证**
  （`mem=0.84`，decode bs=[1] + tail graph 同时捕获，两波 16/16）。
  **DP padding/dead-row/idle 与短尾/whole-extend 的 padded batch 已实现**
  （最大 rank 桶 + `ReqToTokenPool` 预留 slot 0 dummy request + sink SWA
  行；live 行在前、dummy 行在 sink）；A3 真机验证因外部租户占用
  16 die（每 die ~36.8GB）暂未执行；encoder 窗口组合与性能矩阵待补。
  启动/验证脚本已随文档收录在 [scripts/](scripts/)。
- **每 rank 多请求 prefill**：A3 已验证（2026-11-17）：encoder replay
  在 `MAX_RUNNING_REQUESTS=32` / DP16（2 请求/rank）下命中路径
  32/32 正确，`e3_probe 32` FOLD OK 32/32。
- **PD 分离**：NPU 上仍拒绝（P/D 均拒绝，`platforms/npu.py`）。连接器尚未
  消费 `request_window.page_indices()`，窗口页大小、draft 页序与
  `swa_valid_from` 也未验证；验证通过前不放行。
- **A5（M3）**：A3 单机无法验证（需要 arch35 硬件），代码与验证方案待补，
  详见 [03_npu_encoder_design.md](03_npu_encoder_design.md) §8.6 与
  [04_verification.md](04_verification.md) §8。

### 2026-11-19 审查修复（gpt_review2）

针对 `gpt_review2.md` 的 5 个 P1、PD 门控 P2 与设计对齐项已完成修复（细节见
[02](02_npu_decoder_design.md) §9.1、[03](03_npu_encoder_design.md) §8.7、
[04](04_verification.md) §8.8）：

- **分页 SWA 写隔离**：padding/dead 行不再写 `pool.size`（最后一个可分配页
  的首行）；分页 target 池写保留页 0，draft ring 写 ring 外的备用行，
  request window 写自身 sink/workspace scratch，payload 清零且无 host sync。
- **padded batch 类型/新鲜度**：`seq_lens_cpu` 保持 CPU tensor；
  `positions`/`input_ids`/`input_ids_global` 改为图持有缓冲，每次 replay 原地
  刷新。
- **M2 图 key**：加入 padded 请求数；DP 各 rank 先交换该计数，一致才捕获，
  否则全体回退 eager。DeepEP 只允许 low-latency 且桶不超过每 rank 容量。
- **DP 作用域**：捕获与 replay 成对调用 `enter/exit_late_layer_dp`。
- **encoder 窗口**：full/tail metadata 构建与进入/退出 tail 时重新激活对应
  布局，修复 attention 前 `store_cache` 预写使用旧布局的问题。
- **预算**：draft ring 页数与 `create_buffer` 共用 `npu_pool_pages()`，
  不再多预算一页。
- **验证**：`unit_sweep.sh` 15 文件 `NEW_FAILED_FILES=0`（唯一失败为基线已知
  `test_npu_dsv41_low_ratio_backend.py`）；新增
  `test_npu_dsv41_tail_graph_guards.py` 5 例。M2 DP padding/whole-extend 的
  两波真机对照、E3 graph 精度复跑与 PD/A5 仍待设备验证。

### 2026-11-21 A3 重基（dsv41-a3-dev-1112 + PD）+ 精度根因

A3 线已整体重基到用户指定的可工作 DSV4.1 A3 分支
（`HEX1A0A/sglang` `dsv41-a3-dev-1112`，树等价 `54a40d8`），bounded replay
作为独立提交叠加在最上面。提交栈：

1. reference base：A3 适配（树与参考分支逐字节一致）；
2. `fix(npu): allow Ascend DeepSeek V4.1 DSpark PD with CP=1`：来自
   `BourneSun0527/sglang1` `dsv41` 的 PD 分离适配（`ascend` 传输后端 +
   NPU/CP=1 门控 + 配置单测）；
3. `build(npu): tolerate the installed sgl-kernel-npu release wheel`：
   markov_sample softcap 回退、alloc_extend_indices 回退；
4. `[feature][npu][dsv4.1] support bounded replay`：本特性（独立、最新），
   A3 后端与 replay 模块随本提交一起带入。

精度根因：重基树曾缺两处 A3 MoE 语义，参考分支本来就有——

1. `swiglu_limit = 10.0`：A3 W4A8/W8A8 专家必须在 SiLU 前对 gate/up 做
   clip（`_clipped_swiglu_quant`，经 `moe_runner/ascend.py` 传参）；
2. W4A8 w2 `scale_bias` 必须保留 16 段补偿（`bias_last_dim = 16`，
   `_update_bias` 按 MoE TP 窄化后求和）；重基树写成 `16 // tp_size`（=1），
   丢掉 15/16 的 QuaRot 激活移位补偿。

同口径 16 题 AISBench GPQA 探针（gold D、thinking、effort low、1 请求/rank、
DSpark on、replay off）对照：补前 DSpark-on 2/16（DSpark-off 11/16），补后
DSpark-on 11/16(68.75%)；参考分支 13/16(81.25%)。

尾层行桶图：`NpuDecoderReplayTail` 已归档到 `tail_graph/`，不在活动路径。
encoder replay 开启时该图本就不会启用（request window 存在时 `plan()` 返回
False），当前全特性配置走 eager tail；重基后的 A3 树未验证该图路径。

真机验证（参考基座 + PD + 本特性，encoder+decoder replay、DSpark block5、
decode graph bs1/2、radix 全开，TP16/DP16 + DeepEP）：
- E3 命中：wave1 16/16、wave2 16/16；16 rank prefill
  `#cached-token: 1536, #replay-token: 128`；
- 注：上游版 encoder window 布局与参考分支 A3 代码不兼容（E3 复读），
  A3 必须使用本提交内的 replay 栈。

### 2026-11-21 GPQA 长上下文格式排查（结论：跨请求污染，不改 tail）

排查：clean 服务上 all-features（encoder+decoder replay + DSpark + decode
graph）直接跑 doc 配置 2 题 = 2/2（`ANSWER: D`）；同一服务若先跑过 e3 的
16 个 fold 请求，再跑 GPQA 会出现推理连贯但答案全错/只答内容不给字母。
`logprobs: true`（强制 whole-extend）只在该会话内把两题拉回正确，说明
whole-extend 改变的是“已被污染后的表现”，不是根因。

结论：tail（最后 128 行裁剪 + `seq_len−128` floor）本身与 GPU 共享同一套
实现，clean 服务下精度正常；失败来自**跨请求状态污染**——e3 的 fold 请求
结束后，槽位的 request window / `swa_valid_from` 没有重置
（`reset_request_rows` 仅对 fresh 行生效），后续不同请求复用槽位时读到
上一请求的窗口行。临时加的 prefix==0 whole-extend 已回退，保持与 GPU
一致；正确修法是复用/首次写入时重置窗口状态。

另：decoder-only 配置目前无法启动——`late_layer_dp_counts` 的 DP host
all_gather（int64）在 warmup/idle 场景与 4B 集合通信错位
（gloo `op.preamble.length <= op.nbytes. 8 vs 4`）。全特性走 encoder window
路径可绕过，需另行修复。

## 文档索引

| 文档 | 内容 |
|---|---|
| [goal.md](goal.md) | 原始需求与已确认决策 |
| [00_background_and_concepts.md](00_background_and_concepts.md) | **入门必读**：背景、模型事实、tail/whole-extend、bounded replay 名字的含义、与 encoder replay/DSpark 的关系、名词速查 |
| [01_gpu_bounded_replay_architecture.md](01_gpu_bounded_replay_architecture.md) | GPU 两个特性的语义、契约、扩展点、测试覆盖 |
| [02_npu_decoder_design.md](02_npu_decoder_design.md) | decoder replay NPU 设计：M1 eager / M2 图回放 / M3 A5 |
| [03_npu_encoder_design.md](03_npu_encoder_design.md) | encoder replay NPU 设计：E1 eager fold / E2 图 / E3 PD·DSpark |
| [04_verification.md](04_verification.md) | 单测、真机、精度、性能矩阵与验收标准 |

推荐阅读顺序：`00` → `01` → `02` → `03` → `04`。只想快速了解"为什么快、
安全边界在哪"，读 `00` 即可。`00` 中逐段用
【模型事实】/【语义契约】/【GPU 设计】/【NPU 设计】标注，便于区分哪些是
GPU 特有实现、哪些是 NPU 必须同构的语义。
