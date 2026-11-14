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
- **PD 分离**：A3 上 P 侧可启用 encoder replay，D 侧由共享 resolver 拒绝
  encoder replay（D 只按位置接收 P 的窗口状态）；decoder replay + DSpark 的
  P/D 已在 A3 验证（[04](04_verification.md) §8.9）。A5 待真机验证。
- **A5（M3）**：上游 #47660 已合入 A5 支持；本分支 rebase 后 A5 走上游
  `ascend_dsv4_backend.py`，A3 走 fork 的 `ascend_dsv4_backend_a3.py`。
  A5 真机验证矩阵尚未执行，详见 [04](04_verification.md) §8。

### 2026-11-20 上游 rebase（A5 合入后）

- 分支已 rebase 到 `upstream/main`（该基线含
  `843d27c9e2 [Feat][NPU] DSV4.1: Support DeepSeek-V4.1 on Ascend A5 (#47660)`；
  rebase 时 HEAD 为 `e183c415bd`）。本地 A3/A5 合体链不再保留，bounded replay
  提交重新落在上游代码上。
- **双 backend**：A5 使用上游 `hardware_backend/npu/attention/ascend_dsv4_backend.py`；
  A3（arch22）沿用 fork 已验证实现 `.../ascend_dsv4_backend_a3.py`。
  `attention_registry.create_dsv4_backend` 与 `speculative/draft_utils.py` 通过
  `SRTPlatform` 钩子（`get_dsv4_attention_backend_cls` /
  `get_dsv4_draft_backend_cls`）取 backend，NPU 平台内部按 `is_npu_arch35()`
  选择；两者共享同一 KV 池。平台相关分支（backend 选择、V4.1 BF16 行、
  `_kernel_num_heads`、request-window 字节数、replay tail 工厂）都收敛在
  `platforms/npu.py` 的实现里，公共代码只调用钩子。
- **A3 恢复面**：`dsv4_kv_layout.py` 增加 `V41_BF16`（窗口/压缩/index K）；
  pool 的 BF16 写入、pair-state TP padding、indexer 描述符、PD 连续缓冲、
  `DSV4PoolConfigurator` 的 A3 字节/页预算、`_kernel_num_heads` 的 A3
  per-rank 头宽、mHC torch 回退（`dsv4/dsv4_mhc.py`）均已恢复。上游已把
  `npu_kv_layout.py` 改名为 `dsv4_kv_layout.py`。
- **bounded replay**：decoder M1/M2、encoder E1/E3 与 P 侧 PD 逻辑保持不变；
  D 侧继续由共享 resolver 的 `PD decode` 检查拒绝 encoder replay。
- **算子镜像**：A5 直接使用上游的 `kv_compress_epilog_v2` 与
  `sgl_kernel_npu.operators.attention` 最新入口，不再回退到 v1 writer；
  运行前需更新到匹配的 sgl-kernel-npu 镜像（`unit_sweep.sh` 可用
  `KERNEL_PATH` 指向该 wheel）。旧镜像上 `dsv4_memory_pool` 会在导入期
  缺 `kv_compress_epilog_v2`，本地 CPU 校验可临时注入同名 alias。
  `ascend_backend.py` 的 `attention_sinks` 保留旧 wheel 的回退导入（仅
  ImportError 时触发）：当前 A3 验证镜像的 `operators.attention` 尚未导出该
  符号，删除回退会让 NPU backend 无法导入；新镜像上直接导入生效。
- **合并的额外修复**（2026-11-20 patch review）：
  - request window 的 workspace 初始化清零（算子可能扫过整页，未初始化行
    会读出 NaN）；
  - graph 页表宽度按 `max(layout_pages, ceil(max_seq/page))` 计算，长序列
    replay 不再读到 `-1` 哨兵；`init_graph_metadata` 与
    `_refresh_graph_window_page_table` 均传入最终 seq len；replay 每步复用
    缓存的 `p % workspace_pages` 列模板，不再逐步分配页表张量；
  - 捕获前为 A5 draft block 预置一份合法 `c1a_metadata`（旧的零表不是合法
    schedule），replay 仍原地刷新。
- **公共代码收敛**：平台差异统一走 `SRTPlatform` 钩子（`interface.py` 给保守
  默认值，`npu.py` 实现 A3/A5 分派）；`attention_registry`、`draft_utils`、
  `deepseek_v4._kernel_num_heads`、`pool_configurator`、`deepseek_v4_hook`、
  `DecoderReplayTail.maybe_create` 只调用钩子，不再各自判断设备。A3 专属的
  mHC torch 回退收敛为 `_is_npu_a3` 单点，A5 恢复上游融合算子
  （`mhc_pre_dpskv4` / `cann_ops_transformer.mhc_post`）。
- **未合入的 patch 内容**：`("PD decode", ...)` 的删除与
  `SGLANG_DSPARK_DEBUG_ACCEPT`/`SGLANG_DBG_PT` 调试代码；前者与 D 侧继续
  拒绝 encoder replay 的决策冲突，后者是临时诊断。
- **测试基线**：干净 upstream 上 `test_pool_configurator.py` 的 18 个用例
  同样失败，[scripts/unit_sweep.sh](scripts/unit_sweep.sh) 将其列为基线；
  `test_npu_dsv41_low_ratio_backend.py` 在匹配 wheel 下 101 passed。
- **测试**：A3 CPU 套件已按新布局恢复并指向 A3 backend（a3_backend 23、
  a3_cache 20、a3_dspark_cache 3+1709 subtests、a3_dspark_attention 9、
  a3_dspark_draft_attention 13、dspark_metadata_rebase 17、mhc 5）；
  bounded replay 套件（decoder tail 23、request window 22、tail graph guards 5、
  platform gates 14、encoder replay 19、A5 backend 68 等）全绿。A5 backend
  用例已加入 `unit_sweep.sh`（其 `__new__` 构造的后端需要显式给出
  `token_to_kv_pool`）。
- **待办**：A3/A5 真机矩阵（M1/M2/E1/E3、PD、DSpark、精度/性能）在 rebase 后
  尚未重跑；A5 需要匹配的 sgl-kernel-npu wheel；A3/A5 的共享代码去重可在后续
  PR 收敛。

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
