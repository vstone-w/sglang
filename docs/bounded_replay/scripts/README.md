# A3 启动与验证脚本（快照）

本目录是 `/home/w00889861/dsv4/scripts/dsv41` 与开发机上 M2 测试脚本的快照，
便于随设计文档一起查阅。脚本里的绝对路径（模型、kernel、`SGLANG_ROOT`）对应
A3 单机 16 die 环境；换机时用同名环境变量覆盖。

## 环境

- 机器：A3 单机 16 die（arch22），模型 `DeepSeek-V4.1-Flash-1017-w4a8`；
- 代码树：`SGLANG_ROOT=/home/w00889861/dsv4/sglang_bouned_replay/sglang`；
- 配套 kernel：`KERNEL_PATH=/home/w00889861/dsv4/sgl_kernel_npu_2026.11.0`；
- 所有脚本启动前检查端口、模型与 kernel wheel；`run_dsv4.1.sh` 默认
  `KILL_STALE=1`，`validate.sh` 用 `KILL_STALE=0` 避免干扰他人服务。

## 脚本

| 脚本 | 用途 |
|---|---|
| `run_dsv4.1.sh` | 拉起 A3 单机服务（bounded replay / DSpark / radix / 图开关） |
| `validate.sh` | 一键 baseline vs replay：sanity、probe、prefix、GSM8K |
| `validate_loader.sh` | `validate.sh` 副本，仅把 `SERVER_EXTRA_ARGS` 传给 server（loader 线程数） |
| `gsm8k.sh` | sgl-eval GSM8K 精度 |
| `prefix.sh` | generated-shared-prefix 命中率与吞吐 |
| `unit_sweep.sh` | 逐文件 CPU 单测 |
| `run_m2_test.sh` | M2 行桶图冒烟：M2/DSpark/图/请求数与拓扑可覆盖 |
| `m2_probe.py` | M2 两波 probe：wave1 capture、wave2 replay；答案检查 |
| `run_multi_req_test.sh` | encoder replay 每 rank 多请求（32 并发 / DP16） |
| `e3_probe.py` / `e3_hit_accuracy.py` | E3 fold 命中与命中路径精度 |
| `probe_replay.py` / `compare_replay.py` / `summarize_replay.py` | validate 的探针与报告 |

## 已验证配置（A3）

| 场景 | 关键开关 | 结果 |
|---|---|---|
| M2 首版（单请求/rank） | decoder replay、`DISABLE_CUDA_GRAPH=1`、`MEM_FRACTION=0.86`、DSpark off | 两波 16/16 |
| M2 + DSpark aux | 同上、DSpark on | 两波 16/16 |
| M2 + decode graph | `DISABLE_CUDA_GRAPH=0`、`MEM_FRACTION=0.84`、DSpark off | 两波 16/16 |
| encoder 每 rank 多请求 | encoder replay、`MAX_RUNNING_REQUESTS=32`、`MEM_FRACTION=0.90`、DSpark off | 命中路径 32/32，FOLD OK 32/32 |
| M2 DP padding | decoder replay、`M2_REQUESTS=32`、`M2_MAX_RUNNING_REQUESTS=32`、`M2_FILLER_REPS=40`、`MEM_FRACTION=0.86` | 代码已实现；真机验证被外部租户占用阻塞 |

### M2 冒烟命令

```bash
# 单请求/rank、eager decode、DSpark off
M2_DISABLE_GRAPH=1 M2_MEM_FRACTION=0.86 M2_DSPARK=0 \
M2_MAX_RUNNING_REQUESTS=16 M2_REQUESTS=16 \
  ./run_m2_test.sh

# DSpark aux
M2_DISABLE_GRAPH=1 M2_MEM_FRACTION=0.86 M2_DSPARK=1 \
  ./run_m2_test.sh

# decode graph 共存
M2_DISABLE_GRAPH=0 M2_MEM_FRACTION=0.84 M2_DSPARK=0 \
  ./run_m2_test.sh

# DP padding 冒烟（32 并发，非均匀 rank 分布）
M2_DISABLE_GRAPH=1 M2_MEM_FRACTION=0.86 M2_DSPARK=0 \
M2_MAX_RUNNING_REQUESTS=32 M2_REQUESTS=32 M2_FILLER_REPS=40 \
CHUNKED_PREFILL_SIZE=4096 \
  ./run_m2_test.sh
```

`run_m2_test.sh` 的关键环境变量：`M2_DSPARK`、`M2_DISABLE_GRAPH`、
`M2_MEM_FRACTION`、`M2_MAX_RUNNING_REQUESTS`、`M2_REQUESTS`、
`M2_FILLER_REPS`、`M2_MAX_ROWS`、`M2_TP_SIZE` / `M2_DP_SIZE`、
`M2_RUN_SCRIPT`（替换 `run_dsv4.1.sh`，仅限调试）。

## 已知约束

- A3 只支持 `TP16/DP16`（attention TP 1）+ DeepEP；`TP16/DP1` 在
  `_project_wo_a` 因 `n_local_groups=0` 失败，`TP8` 放不下权重。
- `chunked_prefill_size` 会被 DP 除：`4096/16=256`，所以每 rank 每个
  prefill step 的 token 预算很小；需要多请求同 batch 时用更大的
  `CHUNKED_PREFILL_SIZE` 和更短 prompt。
- M2 的显存窗口：`mem=0.80` 时 KV cache 不足；`mem=0.90` 时 tail graph
  capture 后 OOM；`0.84–0.86` 是已验证窗口。
- DeepEP 的 normal prefill dispatch 不可图捕获（内部流同步）；M2 capture
  期间切到 low-latency dispatch，需
  `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`。
- 设备可能被同机其他租户占用：启动前用 `npu-smi info` 检查每 die HBM
  与 process table；外部 16-rank 任务会直接导致加载 OOM。
