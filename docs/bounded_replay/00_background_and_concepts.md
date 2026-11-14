# 背景与基础概念：什么是 Bounded Replay

- 版本：v1（2026-11-13）
- 面向读者：第一次接触 DeepSeek-V4.1 与 SGLang `dsv4` 后端的工程师
- 前置知识：了解 LLM 推理的 prefill/decode 两阶段与 KV cache 即可
- 目的：在阅读 01~04 的设计文档前，先把"为什么需要、叫什么、依赖哪些模型事实"讲清楚

---

## 0. 标注约定：哪些是 GPU 设计

本文混有四类内容，阅读时请按标签区分：

- **【模型事实】**：V4.1 checkpoint/config 固有，与硬件无关（层数、压缩比、
  source 层、窗口 128 等）；
- **【语义契约】**：bounded replay 的算法要求，GPU/NPU 都必须满足，否则结果
  错误（tail/whole-extend 选择、tail source 全量写、窗口 floor、radix 裁窗等）；
- **【GPU 设计】**：当前 GPU 实现的具体机制、参数或路径，**不是语义要求**，
  NPU 可以不同；NPU 对应方案见 02/03；
- **【NPU 设计】**：NPU 方案中的对应选择，标出与 GPU 的差异。

文中在关键段落前用 `【GPU 设计】` 等标签显式标注；第 2.5 节给出完整对照表，
第 7 节的名词表也带类别列。

---

## 1. 问题的起点：prefill 阶段在算什么

LLM 推理分两个阶段：

- **prefill（SGLang 里叫 extend）**：一次把整段 prompt 输入模型，算出每层的
  KV cache，并产出第一个输出 token。长 prompt 的耗时主要在这里（首 token 延迟
  TTFT）。
- **decode**：每步只输入一个新 token，读取已有的 KV cache，产出下一个 token。

Prefill 的计算量随 prompt 长度线性增长：每一层都要对每个 prompt token 计算
attention 和 FFN。对于 128K 级别 prompt，prefill 往往是端到端延迟的大头。

**一个关键观察**：DeepSeek-V4.1 的部分层只读取最近的 128 个位置（滑动窗口），
更早位置的计算结果对这些层是"死数据"。因此这些层在 prefill 时只需要算每个
请求最后 128 个 token，而不是整个 prompt ——这就是 bounded replay 的来源。

---

## 2. 理解它需要的 DeepSeek-V4.1 模型事实

### 2.1 基本结构

**【模型事实】**DeepSeek-V4.1-Flash（本文简称 V4.1）是一个稀疏注意力 MoE 模型：

- 40 个 decoder 层；
- 384 个 routed experts（top-6）+ 1 个 shared expert；
- 每个位置只有一份 **512 宽的 KV latent**，同时作为所有 64 个 query head 的
  key 和 value（MQA/MLA 风格），没有独立的 value 投影；
- 层内除了 FFN/MoE，还有 mHC（多路残差混合）和两个 Engram（n-gram 哈希记忆）
  层；Engram 不在本文关心的 late layers 里。

### 2.2 每层读取的两类 KV

**【模型事实】**V4.1 的注意力不是普通的 full attention，每层同时读两类 KV：

1. **SWA（sliding window attention，滑动窗口）**
   - 每个 token 只关注自己之前最多 128 个位置；
   - 每层用自己的激活重新计算这 128 个位置的 K/V，所以不随上下文增长；
   - 这是 bounded replay 能成立的根本原因。

2. **压缩 latent（compressed KV）**
   - 只在少数 **source 层**产生（V4.1-Flash 为第 2/8/14/20 层，
     `kv_source_layer_ids`）；
   - 第 2~19 层按 2:1 压缩（两个位置合成一个条目），第 20 层起按 1:1；
   - 所有层都可以检索这些压缩条目，所以它们覆盖整个历史，不能只算尾部；
   - 检索时用 indexer 选出 top-512 候选（`index_topk`），再由注意力读取。

### 2.3 late layers（晚期层）与 `late_layer_start`

**【模型事实】**`kv_source_layer_ids = [2, 8, 14, 20]`；
**【语义契约】**`late_layer_start = max(kv_source_layer_ids)`，V4.1-Flash 为 **20**。

- 第 20 层是最后一个 source 层：它要为**所有** prompt 位置写压缩 KV 和
  index K（供后续 decode 检索整个历史）。
- 第 20~39 层（共 20 层）不再产生新的压缩 KV，只做：
  - 自己那 128 位置的 SWA；
  - 对第 20 层发布的压缩 KV / top-k 候选做检索和重打分。
- 因此这 20 层在 prefill 时只需要每个请求的最后 128 个 token（tail）。
  - 这里的“算”指为 tail 行计算**自己的** late-layer 输出和 SWA KV；K/V 是
    该行在该层的 hidden state 投影出来的，不需要更早 token 的 KV。
  - source 层（到第 20 层）仍然对完整 prompt 计算，所以 tail 行进入 late
    layers 的输入 hidden state 是完整可得的。
  - tail 中靠前的行，其真实 SWA 窗口会越过 tail 起点；bounded replay 用
    `swa_replay_start = seq_len - 128` 截断窗口，不补算旧 KV。因此它是
    近似而非 full prefill 的逐位等价（见 §3.7）；需要精确窗口时走
    whole-extend。

### 2.4 为什么 late layers 的旧行可以不算

**【语义契约】**未来 decode 的 query 位于序列末尾，它在这 20 层里只读取"自己之前 128 个位置"
的 SWA KV。prompt 中更早位置的 late-layer SWA KV 永远不会被读到，所以 prefill
不需要为它们计算（也就不会写这一部分 KV）。

fresh prompt 没有旧的 late-layer SWA KV，tail 头部几行的窗口只能截断到 tail
起点；前缀命中时，encoder replay 的 request window 会把旧窗口行从缓存中恢复，
tail 才能读到 tail 起点之前的那段 SWA KV。

注意：第 20 层自己的**压缩 KV** 仍然要覆盖所有 prompt 行，因为压缩路可以检索
整个历史；这是下一节 "tail source KV" 存在的原因。

### 2.5 对照表：模型事实 vs 语义契约 vs GPU 设计

| 内容 | 类别 | 说明 |
|---|---|---|
| 40 层、384 routed experts、512 宽 KV latent、64 query heads | 模型事实 | checkpoint config |
| `compress_ratios`（2:1 / 1:1）、`kv_source_layer_ids=[2,8,14,20]`、`index_source_layer_ids`、`candidate_source_layer_id` | 模型事实 | config 字段 |
| `SWA_WINDOW = 128` | 模型事实 | V4.1 的窗口配置，同时是 kernel/元数据常量 |
| `late_layer_start = max(kv_source_layer_ids)` | 语义契约 | 由 config 推出，GPU/NPU 一致 |
| tail 行 = 每请求最后 `min(128, extend_len)` 行 | 语义契约 | 行选择必须一致 |
| whole-extend 规则（prompt logprob / FULL hidden，DSpark 例外） | 语义契约 | 数值正确性 |
| tail source KV：对完整 extend 写压缩 KV + index K | 语义契约 | 压缩路要覆盖整个历史 |
| CUDA 在压缩步内顺带写 index K | **GPU 设计** | NPU 的 index K 在 index 步写，必须显式补 |
| breakable CUDA graph、图内 eager break、prefill 图内嵌 tail | **GPU 设计** | NPU 只有全图 `NPUGraph`，无 BCG/prefill graph |
| 行桶步长 128、`MAX_ROWS=2048`、`MIN_PREFILL_TOKENS=2048` 环境变量 | **GPU 设计** | NPU M2 复用"行桶"思路，参数另行确定 |
| `swa_replay_start` 元数据字段 | **GPU 设计** | NPU 使用 `dsv41_win_floor` |
| `DecoderReplayTail` 用 `capture_dspark` 作为图 key | **GPU 设计** | 语义只要求 aux 按 tail 行捕获 |
| radix 匹配时裁掉最后 128 token | 语义契约 | NPU 继承同一规则 |
| encoder replay 的 request window（ring + workspace） | **GPU 设计** | NPU E1 用行式窗口（见 03） |
| fold 把 replay 行拼进 extend | **GPU 设计** | NPU 目标路径同构（见 03） |
| GPU 默认开启 decoder replay、NPU 默认关闭 | 部署策略 | 不是模型语义 |
| A3 BF16 1024 B/token、A5 FP8 544 B/FP4 320 B | 硬件事实 | NPU 模块处理 |

---

## 3. 核心概念

### 3.1 tail（尾行）

**【语义契约】**对一个 prefill 请求，late layers 实际要算的行是它 extend 的最后

```text
tail_len = min(SWA_WINDOW, extend_len)      # SWA_WINDOW = 128
```

行。这些行叫 **tail（尾行）**。整个 batch 的 tail 行数 = 每个请求 tail_len 之和。

### 3.2 whole-extend（整段 extend）

**【语义契约】**有些请求必须读 prompt 的每一行，不能只算 tail：

- 请求要 **prompt logprob**（`extend_logprob_start_lens < extend_len`）；
- 请求要返回**每个位置的 hidden state**（`CaptureHiddenMode.FULL`），
  但 **DSpark 投机解码除外**：DSpark 虽然也捕获 hidden state，但只读 tail 行。

这类请求在 late layers 里跑整个 extend，叫 **whole-extend**。它保证 prompt
logprob 等结果的语义正确，也是 bounded replay "数值不逐位等价"说明的一部分。

### 3.3 bounded replay 这个名字

拆成两个词：

- **bounded（有界）**：late layers 的计算行数被限制在窗口大小（128/请求），
  不随 prompt 长度增长；这是语义层面的收益。
- **replay（回放）**：把 late-layer stack 预先捕获成"按行桶的图"，每个 prefill
  步骤回放对应行数的图，降低 kernel launch/调度开销；这是执行层面的收益。

需要注意：

- **【语义契约】**"bounded" 是必须的语义；"replay" 不是语义要求。
- **【GPU 设计】**GPU 有 eager tail（不捕获图）与行桶图回放两条路径；
  **【NPU 设计】**NPU 的 M1 先做 eager tail，M2 才做 NPUGraph 行桶回放。
- 所以文档里有时说 "bounded tail"（只强调语义）和 "bounded replay"（含图回放）。

### 3.4 tail source KV：source 层为什么要先算全量

第 20 层（`late_layer_start`）既是 late layers 的第一层，也是最后一个 source
层。它在 tail 内运行时有两个特殊处理：

1. **先对完整 extend 跑一遍压缩器和 index K 写入**（`write_tail_source_kv`），
   保证压缩 KV/index K 覆盖所有 prompt 行；
2. 记下 `tail_source_kv_layer_id`，尾段内该层再次执行时跳过压缩器
   （`forward_low_ratio_sources` 里的判断），只做检索/打分。

**【语义契约】**"压缩 KV/index K 覆盖所有 prompt 行"是正确性要求。

【GPU 设计】与【NPU 设计】在这里有一个重要差异：

- 【GPU 设计】CUDA 的 index K 在压缩步内写入，所以"跑压缩器"就顺带写了
  index K；
- 【NPU 设计】NPU 的 index K 在 index 步写入（`_dsv41_index_k_store`），
  所以 NPU 的 `write_tail_source_kv` 必须显式调用压缩器 **和** index K 写入，
  否则会静默丢失全量 index K。

### 3.5 行桶（row bucket）与图回放

late layers 的 tail 行数每个 step 都可能不同（请求数、每请求 extend 长度都
会变）。**【GPU 设计】**GPU 用 **行桶** 解决：

- 以 `SWA_WINDOW = 128` 为步长，把行数向上取整到 128 的倍数；
- 每个桶捕获一张图（CUDA 上是 breakable CUDA graph），首次使用时按从大到小
  捕获，后续同桶 step 直接回放；
- 用 `SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS` 限制最大行数（默认 2048），
  0 表示关闭图回放。

【GPU 设计】CUDA 的图里还允许"eager break"（KV 写入、元数据切换等必须每步
重新执行的操作）；【NPU 设计】NPU 的 `torch.npu.NPUGraph` 只有全图，没有
breakable 机制，所以 NPU 的图回放需要把所有每步状态放进固定缓冲、在 replay
前原地刷新（详见 02 的 M2）。

### 3.6 窗口下界（floor）：不是所有旧行都能读

tail 行的位置是绝对位置。一个 tail 行的 SWA 窗口看起来是
`[position-127, position]`，但最早的那个 tail 行可能退到"前一个窗口"之外；
那些位置可能没有 late-layer SWA KV（例如跨 chunk、前缀命中）。因此：

```text
floor = max(0, seq_len - SWA_WINDOW)
```

tail 行的窗口左边界被钳到 `floor`。whole-extend 请求的 floor 不同（窗口不被
截断）。**【语义契约】**floor 公式与钳制行为在 GPU/NPU 上必须一致。
**【GPU 设计】**GPU 把这个下界放在 `swa_replay_start` 元数据里；
**【NPU 设计】**NPU 使用 `dsv41_win_floor`（见 02）。

### 3.7 数值等价性

bounded replay **不是** full prefill 的逐位等价实现：

- 它跳过了 tail 之外 prompt 行的 late-layer 计算；
- 批内其他请求是否 bounded 会影响 batch 形状和 kernel 归约顺序；
- V4.1 本身也声明输出不是跨 batch 组合逐位稳定的。

因此：

- 需要 **prompt logprob** 的请求走 whole-extend（保证读全行）；
- RL 等要求 logprob 与 trainer 一致的场景，应显式关闭（GPU 用
  `--no-enable-decoder-swa-bounded-replay`）；
- NPU 默认关闭、需显式开启，也是出于数值与真机验证的考虑。

---

## 4. Encoder replay：解决前缀复用后的窗口问题

### 4.1 radix cache 与前缀命中

SGLang 用 radix cache 复用相同前缀的 KV：如果两个请求共享一段 prompt，
第二个请求可以只 prefill 新增部分（extend_len 变小，prefix 命中）。

但在 bounded replay 下，late layers 只为每次 prefill 的 **tail 行**写了
SWA KV。前缀命中的最后 128 个位置，其 late-layer SWA 不保证完整（可能来自
更早的、边界不同的 prefill）。所以：

- 匹配前缀时，radix cache 会保守地"裁掉最后 128 个 token"重新 prefill
  （`swa_reprefill_tail_tokens`）；
- 这就是 decoder replay 与 radix cache 的契约。

### 4.2 encoder replay 干的事

Encoder SWA bounded replay 是另一个特性（默认关闭）：

- 命中前缀的 **Main KV 和 Indexer K** 仍然复用；
- 但请求私有的 **SWA 窗口**由重跑"最后 ≤128 个缓存 token"重建。

两种执行方式（都属于执行设计，不是语义要求）：

- **【GPU 设计】fold（CUDA `dsv4`；NPU E1 目标路径同构）**：把这 ≤128 个
  token 拼到 extend 前面，用一次 forward 顺便重建窗口；
- **单独 replay forward（非 fold 后端）**：只构造 replay 行的 batch 再跑一次。

### 4.3 request window（请求窗口）

**【语义契约】**重建的窗口不能写回共享的 radix 页（可能被别的请求复用/覆写）。
**【GPU 设计】**GPU 为每个请求准备一份私有存储（request window）；
**【NPU 设计】**NPU E1 使用行式 request window（见 03）：

- **请求环（request ring）**：每请求、每 SWA 层，保存最近窗口的 KV；
- **workspace**：prefill/fold 的工作区；
- 布局用 `WindowLayout` 描述（历史行、新行、下界、commit 掩码）。
- **两种寻址**：prefill/fold 把历史行+新行 staging 到 workspace（保持
  floor 语义）；decode/verify 用 per-request ring 页表
  （`slot*ring_pages + p%ring_pages`）直接读写请求环，不再 staging。
  实现在 `NPURequestWindowController`。

`floor` 在这里的含义是：replay 行的窗口不能早于该请求的 replay 起点。

### 4.4 两个 replay 的关系

- **decoder replay**：让 late layers 只算 tail、只写 tail 的 SWA KV；
- **encoder replay**：在前缀命中、需要复用旧 KV 时，把 late layers 的窗口补回来。

两者可以叠加：decoder 负责"少算"，encoder 负责"复用后仍正确"。

---

## 5. 一次带 bounded replay 的 prefill（文字流程）

**【语义契约】**下面的主流程（何时切 tail、何时写 source KV、何时恢复）是
GPU/NPU 都必须遵守的；其中"行桶图回放"是可选的执行方式（GPU 用 CUDA 图，
NPU M2 用 NPUGraph）。

```text
请求进入 Scheduler
  |-- radix 匹配；若开启 decoder replay，裁掉前缀最后 128 token
  v
ForwardBatch (完整 extend 行)
  |
  |-- 每个 rank 计算 late_layer_num_tokens（tail 行数），DP 时跨 rank 交换
  v
AttentionBackend.init_forward_metadata
  |-- full metadata（完整 extend）
  |-- tail metadata（只有 tail 行；whole-extend 请求例外）
  v
模型逐层前向：
  |-- 第 start ~ 19 层：完整 extend 行
  |-- 第 20 层（late_layer_start）：
  |     1) write_tail_source_kv：对完整 extend 写压缩 KV + index K
  |     2) enter_late_layer_tail：切到 tail metadata / tail 行
  |-- 第 21 ~ 39 层：只跑 tail 行
  |     （GPU 可选：行桶图回放；NPU M2 可选：NPUGraph 行桶回放）
  |-- exit_late_layer_tail：恢复完整 metadata
  v
logits / hidden states：按 tail 行对齐
  |-- 普通生成：取最后一个 tail 行的 logits
  |-- prompt logprob / FULL hidden states：whole-extend 请求跑全行
  |-- DSpark：aux hidden states 只捕获 tail 行，交给 draft KV 注入
```

---

## 6. 与投机解码（DSpark）的关系

bounded replay 只作用于 **plain prefill（EXTEND）**，不作用于：

- draft 前向（`TARGET_VERIFY`）；
- verify 前向（`TARGET_VERIFY`）；
- decode。

DSpark 的耦合点只有两个：

1. **target hidden capture**：DSpark 需要目标模型某些层的注意力输入；这些
   hidden states 按 tail 行捕获（DSpark 的 draft 层是纯 SWA，只读最后窗口，
   所以只捕获 tail 行是安全的）；
2. **draft KV 注入**：用捕获行的索引把 draft KV 写回；bounded replay 下
   只注入 tail 行。

verify / accept / budget / commit 本身不受影响。

**【GPU 设计】**图回放的 key 里带 `capture_dspark` 属于 GPU 实现细节；
**【语义契约】**语义上只要求 aux hidden states 按 tail 行捕获。

---

## 7. 名词速查

类别说明：**模型** = 模型/checkpoint 固有；**语义** = GPU/NPU 都必须满足；
**GPU** = 仅 GPU 实现；**NPU** = NPU 方案；**硬件** = 设备事实。

| 名词 | 类别 | 含义 |
|---|---|---|
| prefill / extend | 语义 | 一次处理整段 prompt 的前向 |
| decode | 语义 | 每步生成一个 token 的前向 |
| KV cache | 语义 | 每层缓存的 key/value，decode 复用 |
| SWA | 模型 | sliding window attention，只看最近 128 个位置 |
| SWA_WINDOW | 模型 | 窗口长度，固定 128 |
| late layers | 语义 | `late_layer_start`（V4.1 为 20）及之后的层 |
| tail | 语义 | 每个请求 extend 的最后 `min(128, extend_len)` 行 |
| whole-extend | 语义 | 必须跑完整 prompt 行的请求（prompt logprob、FULL hidden） |
| source 层 | 模型 | 产生压缩 KV 的层（V4.1 为 2/8/14/20） |
| tail source KV | 语义 | 第 20 层在完整 extend 上写压缩 KV + index K |
| floor | 语义 | tail 行窗口的左边界 `max(0, seq_len-128)` |
| 行桶 | GPU | 以 128 为步长的行数分桶，用于图捕获/回放 |
| breakable CUDA graph | GPU | CUDA 上允许 eager break 的分段图 |
| NPUGraph | NPU | Ascend 的全图捕获机制（无 breakable） |
| radix cache | 语义 | 前缀 KV 复用结构 |
| 前缀命中 / prefix hit | 语义 | 请求复用已缓存的前缀 |
| request window | GPU | encoder replay 的每请求私有 SWA 环 + workspace；NPU 行式版见 03 |
| fold | GPU | 把命中前缀的最后 ≤128 token 拼进 extend；NPU 目标路径同构 |
| DP attention | 语义 | attention 维度做数据并行，late 行数需跨 rank 交换 |
| PD | 语义 | prefill/decode 分离部署 |
| DSpark | 模型 | V4.1 内置的三阶段投机 draft |
| A3 / A5 | 硬件 | NPU 两代硬件（arch22 / arch35-Ascend 950） |

---

## 8. GPU 设计与 NPU 设计的边界

### 8.1 只属于 GPU 的设计（NPU 不复用）

- breakable CUDA graph + 图内 eager break；
- prefill CUDA graph 内嵌 tail（`run_in_prefill_graph`、bridge buffer）；
- 行桶参数与环境变量：128 步长、`MAX_ROWS=2048`、`MIN_PREFILL_TOKENS=2048`；
- GPU 元数据字段：`swa_replay_start`、`LateLayerTail.scatter_indices`、
  `real_rows_count`、`prefill_graph_tail`；
- `DecoderReplayTail` 的 CUDA 行桶 runner 与 CUDA `_layer_stack`；
- encoder replay 的 packed-page `RequestWindowSpec` / `KVLayout` 与 Triton
  copy/commit kernel；
- GPU 默认开启 decoder replay 的部署策略（NPU 默认关闭）。

注意：下面这些**不是** GPU 独有，NPU 必须同构实现——tail/whole-extend 选择、
tail source 全量 KV/index K、窗口 floor、radix 裁窗、encoder fold 的调度语义。

### 8.2 NPU 的对应差异

GPU 的实现不能直接搬过来，原因在 02/03 文档展开，这里先给出直觉：

- **没有 breakable graph / prefill graph**：NPU 只有全图 `torch.npu.NPUGraph`，
  CUDA 的"图内 eager break"在 NPU 不存在，所以 tail 图要把每步状态做成
  固定缓冲 + 原地刷新；
- **元数据是位置驱动的**：NPU 在 attention 时刻直接读 `forward_batch.positions`
  / `out_cache_loc`，tail 必须同时换 batch 字段，而不只是换 metadata；
- **index K 在 index 步写**：NPU 的 index K 写入与 CUDA 分相不同，tail source
  层必须显式补写全量 index K；
- **行格式不同**：A3 是 BF16（1024 B/token），A5 是 FP8/FP4（544/320 B），
  encoder replay 的请求窗口不能用 CUDA 的 packed-page 结构；
- **A3/A5 算子与 metadata ABI 不同**：A3 固定 2048-int32 metadata，A5 是
  MQSMLA/QLI 的 probe 尺寸。

---

## 9. 建议阅读顺序

1. 本文（背景与概念）；
2. [01_gpu_bounded_replay_architecture.md](01_gpu_bounded_replay_architecture.md)
   ——GPU 两个特性的完整语义与代码索引；
3. [02_npu_decoder_design.md](02_npu_decoder_design.md)
   ——decoder replay 的 NPU 设计（M1/M2/M3）与实现状态；
4. [03_npu_encoder_design.md](03_npu_encoder_design.md)
   ——encoder replay 的 NPU 设计（E1/E2/E3）；
5. [04_verification.md](04_verification.md)——验证矩阵与验收标准。
