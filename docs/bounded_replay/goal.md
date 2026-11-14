# 任务：DSV4.1 bounded replay 的 NPU 适配（A3 → A5）

## 原始需求

> 阅读 `/home/w00889861/dsv4/sglang`。当前已有基于 A3 NPU 平台的 DSV4.1 代码适配，
> 需要继续开发 DSV4.1 的 bounded replay 功能：
> 1. 基于当前 GPU 适配代码，先理解 GPU 整体 bounded replay 适配架构；
> 2. 设计 NPU 适配方案：尽可能复用 GPU 代码，减少公共框架代码修改，修改收敛到
>    NPU 独有模块，减少使用 `is_npu` 分支；文档固化到 `/home/w00889861/dsv4/docs/bounded_replay`。

## 已确认的决策（2026-11-13）

- 范围：**decoder SWA bounded replay + encoder SWA bounded replay 两个都做**。
- 支持面：GPU 支持的部署形态 NPU 都要支持（TP、DP attention 在同等约束下、
  GPU 支持的 PD 角色）；GPU 不支持的（如 context parallel）暂不做。
- 开发顺序：**先 A3（arch22），后 A5（arch35 / Ascend 950）**。
- 默认行为：NPU 上**默认关闭**，通过 `--enable-decoder-swa-bounded-replay` /
  `--enable-encoder-swa-bounded-replay` 显式开启（GPU 默认开启）。
- Decoder 交付节奏：**M1 eager bounded tail 先行，M2 NPU 图回放随后**。
- 代码原则：复用 GPU 框架；公共代码改动最小；NPU 机制收敛在
  `python/sglang/srt/hardware_backend/npu/`；公共代码不新增 `is_npu` 分支，
  用平台能力谓词（`SRTPlatform.supports_*`）/后端虚函数做扩展点。
- 不新增环境变量；沿用 `SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS` 等既有开关。

## 当前状态（2026-11-19）

- **A3 M1（decoder eager tail）**：已实现并真机验证（GSM8K 1.0，
  DSpark accept 3.3–4.7）。
- **A3 E1（encoder eager fold）**：已实现并真机验证（GSM8K 1.0，
  共享前缀命中 47.08%；fold 命中输出用 rank 配对探针校验）。
- **A3 E3 DSpark**：已实现并真机验证（GSM8K 1.0，命中路径 16/16，
  accept 3.76–4.76）；decode/verify 改为 direct ring，TPOT 约 -11%。
- **PD**：未支持；NPU 上 P/D 两个角色均被 resolver 拒绝，待连接器消费
  `request_window.page_indices()` 并验证窗口页大小/页序/`swa_valid_from`。
- **NPU graph（encoder decode/verify）**：HCCP 初始化修复后已解锁；
  A3 无 replay 与 E3 target verify graph capture 通过，resolver 不再
  要求 `--disable-cuda-graph`；E3 graph 完整精度复跑待做，draft ring
  的图捕获已打开（真机验证待补）。
- **M2（decoder prefill tail 行桶图）**：首版已实现并在 A3 验证
  （单请求/rank、128 行桶、DSpark aux、decode graph 共存）；
  DP padding/whole-extend 的 padded batch 已实现，2026-11-19 完成
  gpt_review2 的 5 个 P1 修复，真机两波对照待设备。
- **A5（M3）**：未开始。
- **2026-11-19 审查修复**：分页 SWA sink 隔离、padded batch 类型/输入
  刷新、图 key 请求维度、DeepEP 模式/容量门限、DP 作用域配对、encoder
  窗口激活、窗口预算页数、PD 门控已在代码与 CPU 单测落地；详见
  `02` §9.1、`03` §8.7、`04` §8.8。
- 代码结构：窗口逻辑收敛在
  `hardware_backend/npu/dsv4/npu_request_window.py` /
  `dsv4/dspark_ring.py`；tail 行桶图已归档到
  `docs/bounded_replay/tail_graph/`（encoder replay 开启时本就不启用，
  当前验证配置走 eager tail）；共享 resolver 无新增 `is_npu` 分支。

待办与验证矩阵见 `03_npu_encoder_design.md` §8.7 与
`04_verification.md` §8.8。

## 文档索引

- `README.md`：术语、数据流总览、文档索引。
- `00_background_and_concepts.md`：背景、模型事实与 bounded replay 基础概念（入门必读）。
- `01_gpu_bounded_replay_architecture.md`：GPU 两个 bounded replay 特性的完整架构基线。
- `02_npu_decoder_design.md`：decoder replay 的 NPU 设计（M1/M2/M3）。
- `03_npu_encoder_design.md`：encoder replay 的 NPU 设计（E1/E2/E3）。
- `04_verification.md`：单测 / 真机 / 精度 / 性能验证矩阵与验收标准。
