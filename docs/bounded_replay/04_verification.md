# Bounded Replay NPU —— 验证方案与验收标准

- 版本：v1.2（2026-11-19，审查修复与回归）
- 适用：[02_npu_decoder_design.md](02_npu_decoder_design.md) 的 M1/M2/M3、
  [03_npu_encoder_design.md](03_npu_encoder_design.md) 的 E1/E3
- 原则：先用 CPU 单测钉住契约，再用 A3 真机做 parity/精度/性能；A5 复用同一矩阵

---

## 1. 测试分层

| 层 | 运行环境 | 目的 | 放置 |
|---|---|---|---|
| U-CPU | CPU，AST 抽取 + mock | 钉住纯逻辑契约（布局、视图、保存/恢复、guard） | `test/registered/unit/` |
| U-NPU | A3 单卡，真实 torch_npu | 钉住行格式、图刷新、窗口 gather/commit | `test/registered/npu/spec/`、`test/registered/unit/npu/` |
| E2E-A3 | A3 单机（TP/DP/PD） | 端到端正确性、精度、性能 | `test/registered/npu/accuracy/`、`performance/` |
| E2E-A5 | arch35 单机 | 同上，A3/A5 对照 | 同上，suite 换 A5 |

NPU 单测遵循现有模式（`test/registered/unit/npu/attention/`）：
AST 抽取生产代码、`exec` 到隔离命名空间、mock `torch_npu`/`sgl_kernel_npu`、
用 lambda 切换 `is_npu_arch35` 并互设哨兵。真机测试用
`register_npu_ci(..., suite="nightly-1-npu-a3", nightly=True)`。

## 2. Decoder M1 测试

### 2.1 U-CPU

| 文件 | 用例 | 防住的回归 |
|---|---|---|
| `test/registered/unit/layers/test_dsv41_late_layer_tail_readers.py`（扩展） | tail 布局 bs=1 / bs>1、short×window、whole-extend 混合、非均匀 seq_lens；行顺序与 floor | 共享 helper 改动导致行/floor 错位 |
| 新增 `test/registered/unit/model_executor/test_dsv4_tail_batch_view.py` | tail ForwardBatch 视图：positions/out_cache_loc/input_ids gather、extend lens、prefix lens、start loc、seq_lens 不变 | 漏换字段（位置驱动元数据的静默错读） |
| 新增（CPU） | `enter/exit_late_layer_tail` 保存/恢复全集；异常路径也恢复 | 状态泄漏到后续 batch |
| 新增（CPU） | `_low_ratio_index_topk` K-store guard：tail source -> no-op；无标记 -> raise | 漏写 index K 或静默覆盖 |
| 新增（CPU） | resolver：NPU 默认关闭、显式开启走兼容检查、checkpoint 字段缺失报错 | 启动崩溃或错误默认值 |

### 2.2 U-NPU（A3）

| 文件 | 用例 | 防住的回归 |
|---|---|---|
| `test/registered/npu/spec/test_npu_dsv41_tail_source_kv.py` | `write_tail_source_kv` 对 2 请求 extend：压缩行全写、index K 全写、无 topk 写 | NPU 分相写入契约 |
| `test/registered/npu/spec/test_npu_dsv41_tail_metadata.py` | tail 元数据：`actual_seq_lengths_q_pa`、`start_pos`、`seqused`、`swa_loc`、`c1_loc` 与手算一致 | 元数据构造错误 |
| `test/registered/npu/spec/test_npu_dsv41_swa_tail_slots.py` | `get_swa_out_cache_loc` 返回 tail 槽，`store_cache` 不越界 | tail KV 写错槽 |

### 2.3 E2E（A3）

前置：确认 A3 checkpoint `config.json` 含 `kv_source_layer_ids`、
`index_source_layer_ids`、`candidate_source_layer_id`、`compress_ratios`；
缺失时启动即报错。

| 场景 | 检查 |
|---|---|
| 长 prompt（~4k）贪心，feature on/off | 首批 token 一致；记录 KL/精度差异（bounded replay 非数值等价） |
| prompt logprob 请求 | 与 full prefill **逐位一致**（whole-extend） |
| 短 prompt（<128、=128、=129） | 与 full prefill 一致（tail 即全量或跨窗口） |
| chunked prefill（chunk 128/256/512/2048） | 不跨 chunk 错位 |
| bs=1 / bs=8 非均匀长度 | 无跨请求污染 |
| radix 前缀命中 | 最后 128 token 被 reprefill（`cached_tokens` 相应减少） |
| DSpark | accept length > 1；draft KV 注入行正确；与 feature off 无明显回退 |
| DP attention | **M1 先拒绝**；M2b 起按 `moe_a2a_backend == "none"` 验证 2/4/8 卡 |
| PD | P/D 角色分别验证；未通过前 resolver 拒绝 |

精度：使用 `sgl-eval`（项目规则），GSM8K 与 CUDA 同等阈值；
若模型有 `--load-preset-from-model-id` 预设则使用；采样参数按模型推荐设置。
报告时必须注明 harness 与参数。

## 3. Decoder M2 测试

| 文件 | 用例 | 防住的回归 |
|---|---|---|
| `test/registered/npu/spec/test_npu_dsv4_tail_graph_parity.py` | eager tail vs graph tail：残差/pre/aux 逐元素一致；桶边界 1/128/129/256/max_rows；连续 replay 输入变化 | 固定地址读到旧数据（NPU replay 经典故障） |
| `test/registered/npu/spec/test_npu_dsv4_tail_graph_fallback.py` | whole-extend / prompt logprob / 非可捕获 MoE 计数域 -> eager，不捕获 | 静默把 whole-extend 塞进 128 行桶 |
| `test/registered/npu/spec/test_npu_dsv4_tail_graph_idle.py`（2/4 卡） | DP 下 idle rank 回放同桶，集合通信完成 | 行数不对称导致 hang |
| `test/registered/npu/spec/test_npu_dsv4_tail_graph_capture.py`（2/4 卡） | 所有 rank 捕获相同桶集合（从大到小），对称 | 捕获期 HCCL 不对称 |
| e2e | 同 M1 场景 + 图开启 | 图/ea ger 数值漂移 |

性能：记录捕获耗时/显存、图命中率、桶分布；
`SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS` 扫描（256/512/1024/2048）。

## 4. Encoder E1 测试

### 4.1 U-CPU

| 文件 | 用例 | 防住的回归 |
|---|---|---|
| 新增 `test/registered/unit/mem_cache/test_dsv41_window_layout_torch.py` | 纯 torch `window_layout`：num_groups、floor、padded_rows、pos 跳变、单行请求；与手算 indices/lengths/history 对照 | 唯一 NPU 路径的布局回归 |
| 新增 | fold 元数据：keep_rows/row_floor/replay_lens；replay 行压缩写槽置 0 公式 == CUDA `_compressor_out_loc` | 覆盖缓存压缩 KV/index K |

### 4.2 U-NPU（A3）

| 文件 | 用例 | 防住的回归 |
|---|---|---|
| `test/registered/npu/spec/test_npu_dsv41_request_window_rows.py` | `NPURequestWindow` activate/buffer/commit/reset：历史行拷贝、新行写 `req*capacity+pos%capacity`、tag 正确 | 行 gather/commit 错误 |
| `test/registered/npu/spec/test_npu_dsv41_encoder_attention.py` | 单层：fold 后窗口注意力 == 对同一窗口的全量重算 | **A3 恒等页表假设的 go/no-go** |
| `test/registered/npu/spec/test_npu_dsv41_ratio2_replay.py` | ratio-2 压缩器在 fold 下重跑与原始写入逐位一致 | 窗口重建污染压缩状态 |

### 4.3 E2E（A3）

| 场景 | 检查 |
|---|---|
| 相同前缀重复请求（radix 命中） | 贪心输出与关闭 encoder replay 的全量重算一致；`cached_tokens > 0` |
| 多轮对话前缀增长 | 每轮窗口正确重建 |
| 混合命中/未命中/不同请求 | 无跨请求窗口污染 |
| `swa_gates_prefix_match` | 窗口存在时必须为 False |
| 与 decoder replay 同时开启 | 组合路径一致；prompt logprob 请求 whole-extend |
| 非 fold 路径（若保留） | 单独 replay forward 正确 |

## 5. A5（M3）验证

- 先跑 A3 全矩阵，A5 复用同一用例；
- 额外检查：64 query heads 约束、MQSMLA/QLI metadata、fp8/fp4 行格式、
  DSpark 稀疏索引为物理 SWA 槽；
- 输出 A3/A5 对照表（精度、TTFT、吞吐、捕获成本）；
- A5 当前对 TP 尾部 verify 行报错，需确认 DSpark tail 是否触发。

## 6. 性能验收

| 指标 | 方法 | 目标 |
|---|---|---|
| 长 prompt prefill TTFT | feature off vs on，相同并发/长度 | 明显下降（按实测记录） |
| 端到端吞吐 | GSM8K/固定数据集 | 不低于 feature off |
| M2 图收益 | eager tail vs graph tail | 调度开销下降 |
| 图捕获成本 | 启动时间/显存 | 可接受，`MAX_ROWS` 可调 |
| Encoder 命中率 | `cached_tokens`/prefix 命中 | 命中时应减少重算 |

测量必须使用 cookbook 推荐的部署配置（GPU 侧参考
`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`；NPU 侧使用 A3/A5
对应 recipe），并在报告中注明与 cookbook 的差异。

## 7. 退出标准

| 阶段 | 退出标准 |
|---|---|
| M1 | U-CPU/U-NPU 全绿；A3 e2e 长/短 prompt、prompt logprob、chunked、bs、radix、DSpark 通过；性能记录完成 |
| M2 | eager/graph parity 全绿；桶边界、连续 replay、DP idle、对称捕获通过；图收益记录 |
| E1 | 单层窗口 parity 通过；前缀命中 e2e 全绿；ratio-2 幂等；与 decoder 组合通过 |
| E3 | PD P/D 与 DSpark 窗口/草案链路真机通过（未通过前 resolver 拒绝） |
| M3 | A3 全矩阵在 A5 复跑通过，A3/A5 对照表入档 |

截至 2026-11-19 的实际执行状态见 §8（含 §8.8 审查修复回归）；E3 的
DSpark 部分已通过，PD 仍待办。

## 8. 已执行验证（2026-11-19，A3）

环境：A3 单机 16 die，`TP16 + DP16`（attention TP 1）、
`--moe-a2a-backend deepep`、`--disable-cuda-graph`（eager decode）、
`--mem-fraction-static 0.88`（DeepEP HCCL 缓冲约 5.2GB）。
精度 harness：sgl-eval CLI（thinking off），GSM8K 50 题。

### 8.1 结果汇总

| 场景 | 结果 | 备注 |
|---|---|---|
| M1 decoder eager tail | GSM8K 1.0000 | DSpark accept 3.3–4.7 |
| E1 encoder eager fold | GSM8K 1.0000 | 共享前缀 32 条命中 47.08%（encoder off 0%） |
| E1 fold 命中输出 | 16/16 正确 | `e3_probe.py` rank 配对；fold 为近似，不要求逐位一致 |
| E3 encoder + DSpark | GSM8K 1.0000 | accept 3.76–4.76 |
| E3 命中路径 GSM8K | 16/16 正确 | `cached=1536, replay=128`（`e3_hit_accuracy.py`） |
| E3 共享前缀 | 47.08% 命中 | 与 E1 相同；accept 3.76–4.76 |
| 非 replay DSpark eager | 通过 | draft 池 `storage=paged`，事实型 prompt 正确 |
| A3 无 replay graph（DSpark） | target verify capture 通过 | HCCP 修复后 ~11.7s，无 507018 |
| E3 encoder replay + graph | target verify capture 通过 | 首次 fold 探针在 idle replay 处崩溃，已加 commit 保护；完整精度待复跑 |
| 严格 idle 内存检查 | 无泄漏告警 | `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` 开启 |

### 8.2 decode/verify direct ring A/B

同机同配置，direct ring vs workspace staging：

| 指标 | staging | direct | 变化 |
|---|---|---|---|
| GSM8K 50 | 96.2 s | **84.9 s** | -11.7% |
| prefix 32 条（32K+4K, 256 output） | 234.0 s | **218.3 s** | -6.7% |
| 输出吞吐 | 35.0 tok/s | **37.5 tok/s** | +7.2% |
| TPOT | 210.7 ms | **187.0 ms** | -11.2% |
| cache 命中率 | 47.08% | 47.08% | = |

direct 的命中路径 16/16、fold 探针 16/16、GSM8K 1.0000 均通过。
单层 parity（隐式窗口 + 模环页表，L=1/64）max diff ≈ 0.0035。

### 8.3 NPU graph（2026-11-17，A3）

- **根因/修复**：A3 `SparseFlashMlaMetadata`（AICPU）在 custom AICPU
  dispatch 之后才启动 HCCP 时会报 507018。`ensure_hccp_initialized` 在
  `_init_dsv4_graph_buffers` 最前面做一次 TP all-reduce + sync，早于第一个
  metadata seed；graph backend 也调用同一幂等 helper。
- **A3 无 replay**：target verify capture 通过（~11.7s），resolver 不再
  强制 `--disable-cuda-graph`。
- **E3 encoder+DSpark**：target verify graph capture 通过；draft ring 的
  捕获门控已打开（`_refresh_graph_dspark_sparse_metadata` 在 replay 前
  原地刷新稀疏索引/ring 页表，CPU 契约测试覆盖 `swa_valid_from` 与 idle）。
  首次 fold 探针在 DP idle replay 触发 0 长度 mask 拷贝，已在
  `refresh_graph` 加 idle 保护（保留 dummy layout，commit_mask 清零）；
  E3 graph 与 draft ring graph 的完整 fold/命中/GSM8K 待真机复跑。
- **仍未覆盖**：decoder M2 prefill tail graph、E3 graph 性能对照。

### 8.4 单测

- `unit_sweep.sh`：14 文件，13 绿 + 1 个基线已知失败
  （`test_npu_dsv41_low_ratio_backend.py` 在干净基线同样失败）。
- 本次新增/扩展：`test_npu_dsv41_decoder_tail.py`（21）、
  `test_npu_dsv41_request_window.py`（15）、
  `test_bounded_replay_platform_gates.py`（11）、
  `test_dsv41_dspark_metadata_rebase.py`（17）等。

### 8.5 尚未执行/待补

- chunked prefill 的固定 chunk 尺寸矩阵（128/256/512/2048）、bs>1 非均匀长度；
- 每 rank 多请求的 prefill 窗口（当前 resolver 限制
  `max_running_requests <= attn_dp_size`；decode/verify 的 ring 页表已
  天然按请求寻址）；
- PD P/D、A5（M3）、M2 尾层图、graph 解锁；
- 完整性能矩阵（TTFT、吞吐、图收益、捕获成本）与 cookbook 差异说明。

### 8.6 组合路径与 prompt-logprob 逐位一致（2026-11-15，A3）

配置：decoder + encoder + DSpark，TP16/DP16 + DeepEP、eager、
`--mem-fraction-static 0.90`、strict idle mem check 开。
运行目录：`scripts/dsv41/outputs/combined2_20261115_090756`。

| 检查 | 结果 |
|---|---|
| prompt logprob（`logprob_start_len=0`，fresh 4267-token prompt） | baseline vs decoder-only：**逐位一致**（4266 行，max_abs_diff 0.0）；baseline vs combined：同样 **逐位一致** |
| E3 fold 探针（16 rank 配对） | 16/16 命中正确，且命中与 fresh **逐位相同**（cached=1408/1511） |
| 命中路径 GSM8K（wave1/wave2 各 16） | 16/16 + 16/16 正确 |
| 共享前缀 32 条（32K system） | 命中 **47.08%**（567936/1206263），DSpark accept ≈4.29 |
| GSM8K 50（sgl-eval，thinking off） | **1.0000** |
| 事实型 prompt / 泄漏 | 正确；无异常、无泄漏告警 |

组合路径首次上机时暴露并修复了两个缺陷（都已加 CPU 回归用例）：

1. **decoder replay floor 未传入 request window**：late layers 只算 tail，
   但窗口布局仍按 `encoder_swa_row_floor`（无命中时为 0）staging history，
   会读取 late layers 从未写过的环行 → NaN/vector core 异常。修复：tail 视图的
   `encoder_swa_row_floor = swa_replay_start`（decoder floor 与 fold floor 的合并值）。
2. **fold 的 `encoder_swa_compress_rows` 未重映射到 tail 行**：该索引指向
   折叠后的完整 extend，却被用于 gather 只有 tail 行的 `out_loc` → 越界
   （错误在异步后表现为 `aclnnNonzeroV2`）。修复：`_tail_compress_rows()`
   把它映射为 tail 行内索引。

### 8.7 M2 行桶图与每 rank 多请求（2026-11-18，A3）

测试脚本快照见 `docs/bounded_replay/scripts/`。

| 场景 | 配置 | 结果 |
|---|---|---|
| M2 首版（1 请求/rank） | decoder replay、`DISABLE_CUDA_GRAPH=1`、`mem=0.86`、DSpark off、`SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=2048` | wave1（capture）**16/16**、wave2（replay）**16/16**；capture 0.81–0.97s |
| M2 + DSpark aux | 同上、DSpark on | wave1/wave2 **16/16**；`mhc_stream_mean` aux 随图捕获 |
| M2 + decode graph | `DISABLE_CUDA_GRAPH=0`、`mem=0.84`、DSpark off | decode graph `bs=[1]` 5.7GB + tail graph 0.81s 同时捕获；wave1/wave2 **16/16** |
| M2 on/off 逐 token | 1 请求/rank | 32 条中 30 条相同；2 条 greedy 近似翻转（均含正确答案） |
| encoder 每 rank 多请求 | encoder replay、`MAX_RUNNING_REQUESTS=32`、`mem=0.90`、DSpark off | 命中路径 wave1/wave2 **32/32**；`e3_probe 32` FOLD OK **32/32**（16/32 逐位相同） |
| M2 DP padding（32 并发） | decoder replay、`MAX_RUNNING_REQUESTS=32`、`mem=0.86` | 代码已实现并完成 review2 P1 修复；设备被外部 16-rank 租户占用（每 die ~36.8–60.9GB），真机验证待设备空闲 |

**M2 关键契约/修复**

- A3 只有全图 `NPUGraph`，metadata 必须在图外重建再原地拷入；所有
  `*_page_table` 按池最大页宽 padding。
- DeepEP **normal prefill dispatch 内部有流同步，不可图捕获**；capture 期间
  临时置 `is_extend_in_batch=False` 走 **low-latency dispatch**，并设
  `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`。
- DP padding：以所有 rank 最大 live 行数取 128 对齐桶；live 行在前，缺失行用
  `ReqToTokenPool` 预留 **slot 0** dummy request 补齐；dummy 的 SWA 写槽为
  `-1`。**分页 target 池的 sink 是保留页 0**（写 `pool.size` 会命中最后一个
  可分配页的首行，覆盖真实 KV）；draft ring 用 ring 外的备用行，request
  window 用自身 sink/workspace scratch。不可依赖 Python 负索引。
- 短尾/whole-extend 通过 padded batch 的真实 tail lens 与
  `extend_logprob_start_lens_cpu` 标记处理；idle rank 走全 dummy 桶参与集合通信。
- 显存窗口：`mem=0.80` 时 KV cache 不足；`mem=0.90` 时 tail graph capture 后
  OOM；`0.84–0.86` 为已验证窗口。

**单测**：`unit_sweep.sh` 15 文件，**0 个新增失败**（`NEW_FAILED_FILES=0`）；
唯一失败为已知基线 `test_npu_dsv41_low_ratio_backend.py`。

### 8.8 审查修复回归（2026-11-19，CPU）

针对 `gpt_review2.md` 的修复与对应回归：

| 审查项 | 修复 | 回归 |
|---|---|---|
| P1 分页 SWA padding 覆盖真实 KV | 分页 target 池 sink 改为保留页 0；draft ring 用备用行；window 用自身 sink/scratch；dead payload 清零；删除 fused 入口重复映射 | `test_npu_dsv41_tail_graph_guards.py::test_swa_sink_*`、`::test_dead_rows_*`；`test_dsv41_a3_cache.py` |
| P1 `seq_lens_cpu` 类型 | padded batch 恢复 CPU tensor；`extend_seq_lens_cpu` 保持 list | M2 padded batch 暂无独立 CPU harness，需真机/后续补测；M1 tail 测试覆盖同一 metadata 调用链 |
| P1 同形状 replay 读旧输入 | `positions`/`input_ids`/`input_ids_global` 图持有缓冲，replay 原地刷新 | `::test_replay_refreshes_positions_and_input_ids` |
| P1 同一行桶不同请求维度 | 图 key 加 padded 请求数；DP 各 rank 交换计数后一致才捕获 | `::test_padded_request_count_*` |
| P1 DeepEP 模式/容量 | `_moe_can_capture`：仅 none/DeepEP v1/v2 low-latency 且桶 ≤ 容量 | `::test_moe_capture_*` |
| P2 PD 门控与测试冲突 | 恢复 NPU PD（P/D）拒绝，待连接器消费 `page_indices` | `test_bounded_replay_platform_gates.py` |
| P2 窗口预算多一页 | 预算与分配共用 `npu_pool_pages()` | 同文件 `test_npu_request_window_nbytes_uses_row_layout` |
| 设计项 DP 作用域 | 捕获与 replay 成对 `enter/exit_late_layer_dp` | 无纯 CPU 用例；真机 DP 矩阵待补 |
| 设计项窗口激活 | init 构建 tail 后重绑 full；enter/exit 绑定 tail/full | `test_npu_dsv41_decoder_tail.py::test_enter_exit_*` |
| 设计项 D2H | `_tail_compress_rows` 去掉 `int(rows.max())` | `test_npu_dsv41_decoder_tail.py` |

验证结果：

- `unit_sweep.sh`：15 文件 `NEW_FAILED_FILES=0`，唯一失败为基线已知的
  `test_npu_dsv41_low_ratio_backend.py`；
- 新增 `test_npu_dsv41_tail_graph_guards.py` 5 例，AST 抽取生产代码，
  覆盖请求维度、MoE 门控、输入刷新、SWA sink、dead 行清零；
- 针对性复跑：`test_dsv41_a3_cache.py` + `test_dsv41_a3_dspark_cache.py`
  23 passed / 1715 subtests；`test_npu_dsv41_request_window.py` 21 passed；
  `test_bounded_replay_platform_gates.py` 14 passed。
- `py_compile` 与 `git diff --check` 通过。

仍未执行：M2 DP padding / whole-extend 的两波真机对照、E3 graph 完整精度
复跑、DSpark draft ring graph 真机验证、PD P/D、A5 全矩阵。公共
`DecodePreallocQueue` 的 SWA 准入调整保留，需要覆盖原有 GPU/NPU SWA PD
部署的独立验证；`build_prefill_layout`/`build_verify_layout` 的
`int(spans.sum())`、递归 metadata 拷贝的显式化列为后续性能项。

### 8.9 PD + DSpark 与 decode graph（2026-11-19，A3）

**门控放开**：`arg_groups/deepseek_v4_hook.py` 原先要求 DSpark PD 的传输后端
必须是 Mooncake/NIXL/MoRI，`ascend` 被直接拒绝。Ascend 连接器已具备 draft KV
传输（`num_draft_entries`、`draft_kv_layers`），且运行时握手会校验两端
block size、target/draft KV layout 与 TP size，因此把 `ascend` 加入允许列表。
P/D 两端都打补丁后，PD + DSpark + decoder replay + decode graph 冷启动通过。

**decode graph 捕获失败的根因**：`aclnnMoeLowLatencyDispatchV2 ... Failed to
execute tiling function` 来自 DeepEP low-latency workspace 过小。修复是把
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 提到 **256**（至少覆盖一个
128 行 tail 桶），`DEEPEP_HCCL_BUFFSIZE` 用 **2500**，`mem-fraction-static`
用 **0.84**。`96 / 512` 的组合会在 capture 阶段失败。

| 场景 | 配置 | 结果 |
|---|---|---|
| 单机 decoder replay + DSpark + decode graph | TP16/DP16、deepep、mem 0.84、dispatch 256、HCCL 2500 | decode graph（target/draft verify bs=[1,2]）捕获通过；tail graph 捕获 128 行 0.81s |
| PD P/D + DSpark | P: decoder replay、`--disable-cuda-graph`、mem 0.90；D: decoder replay、`--cuda-graph-bs-decode 1 2`、mem 0.84、dispatch 256、HCCL 2500 | 两端 healthy；router 请求 200；D `npu graph: True`、`accept len 2.05` |
| PD + encoder replay | P 端加 `--enable-encoder-swa-bounded-replay`，D 端 decoder replay + DSpark + decode graph | 两端 healthy；24 条相同 3.6k prompt 请求全部正确；DP1–DP10 命中 `cached=3584`，fold 路径生效 |

**PD + encoder replay 的实现**：`platforms/npu.py` 的 PD 拒绝已移除。
P 侧 `resume_units` 在 `request_window` 存在时调用
`NPURequestWindow.page_indices(slot, begin, end)`，按位置升序发出窗口环行；
D 侧同一函数返回 paged SWA 页号，两端按位置一一对应。池的
`get_state_buf_infos()` 把 `request_window.state` 注册为 SWA 池，
item 字节按 `swa_page_size // window.page_size` 换算。

**精度（2026-11-19，PD + encoder/decoder replay + DSpark + decode graph）**：
GPQA-Diamond 10 题，`max_tokens=32768`、`top_p=0.95`、`reasoning_effort=max`、
batch 1、stream。结果 **9/10 = 90%**；平均输出 9174 token、TTFT 661 ms、
TPOT 30.6 ms、28.62 tok/s。唯一错题（index 8）输出 126k 字符且无
`ANSWER:` 行，是 32768 token 截断，不是 replay 精度回退。其余 9 题均正常
给出答案，其中 index 7 用了约 27k token。

已知限制：tail graph 捕获要求所有 attention-DP rank 的 padded 请求数一致；
warmup 只有部分 rank 有请求时会退回 eager（不报错）。
