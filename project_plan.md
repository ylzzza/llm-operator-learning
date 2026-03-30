# vLLM KV Cache INT8 量化项目计划

更新时间：2026-03-25

## 项目名称

基于 `vLLM` 的大模型推理 `KV Cache INT8` 量化与 `Paged/Decode Attention` 算子改造

## 一句话目标

围绕大模型推理 `decode` 阶段的显存带宽瓶颈，在 `vLLM` 现有 `KV cache` 量化链路基础上，扩展一条 `INT8 KV cache` 路径，并完成：

- `KV` 特征分布分析
- `FP16` 高精度参考算子
- `INT8` 量化算子实现
- `vLLM` 底层 attention 分支接入
- 端到端精度与性能评估

最终目标不是“做一个实验脚本”，而是交付一条可以稳定复现、可量化评估、可写进简历的完整工程链路。

---

## 1. 背景与问题定义

### 1.1 问题背景

在大模型推理中，`decode` 阶段通常是典型的 `memory-bound` 场景。随着上下文增长，`KV cache` 的读写与访存会逐步成为主要瓶颈：

- `prefill` 更偏算力密集
- `decode` 更偏 `KV cache` 带宽受限
- 模型越大、上下文越长，`KV cache` 带来的显存占用和访存成本越高

因此，`KV cache` 量化的目标有两个：

1. 降低 `KV cache` 显存占用
2. 减少 `decode` 时的访存带宽压力

### 1.2 当前 vLLM 状态

从当前本地源码和官方文档看，`vLLM` 已经支持：

- `FP8 KV cache`
- `W8A8 / FP8` 线性层量化

但**没有现成的 `INT8 KV cache` 官方主线实现**。因此这个项目的本质不是“调一个已有开关”，而是：

- 基于现有 `kv_cache_dtype + k_scale + v_scale` 接口
- 扩展一条新的 `INT8 KV cache` 数据类型与算子路径

### 1.3 关键约束

- 不能一上来做“全 backend、全模型、全平台”支持
- 独立开发应先打通一条最短可验证路径
- 动态 `per-channel` 很可能带来过大的 scale 元数据开销
- 因此推荐采用：

`动态 per-tensor / per-head MVP -> 静态 calibrated per-channel 成品`

---

## 2. 项目范围

### 2.1 In Scope

- `decode` 阶段 `KV cache` 量化
- `INT8 symmetric quantization`
- `vLLM` 注意力路径接入
- 单机单卡优先
- 单模型优先
- `Qwen2.5-7B` 或 `Llama-3.x-8B` 作为主验证模型
- `operator-level + integration + end-to-end` 三层验证

### 2.2 Out of Scope

- 多机分布式优化
- 多 backend 同时首发支持
- 训练阶段量化
- `prefill` 主路径重写
- 全模型权重量化框架设计

---

## 3. 推荐技术路线

## 3.1 总体路线

建议按下面的顺序推进：

1. 跑通 `vLLM` 基线并建立测试环境
2. 写 `PyTorch FP16` 等效参考算子
3. 做 `KV cache` 分布分析与离线量化模拟
4. 实现 `INT8 MVP`
5. 接入 `vLLM` 指定 attention backend
6. 做静态 `per-channel` 演进
7. 做端到端评测与结果沉淀

### 3.2 推荐先支持的实现路径

独立开发建议优先选择**一条 backend 路线**，不要一开始铺太开。

推荐顺序：

1. `FLASH_ATTN` 路线
2. `TRITON_ATTN` 路线
3. `FLASHINFER` 后续再补

推荐原因：

- `FLASH_ATTN` 在 `vLLM` 中是重要主线路径
- 已有 `FP8 KV cache` 相关测试样板
- 更适合作为“简历项目”的主战场

但具体落地时，不要一开始依赖外部库内部实现来 debug 全部问题。更稳的做法是：

- 先写你自己的 `FP16` 参考实现
- 再写你自己的 `INT8` 参考实现
- 最后接入 `vLLM` 的 backend 调度层

---

## 4. 项目分阶段计划

## Phase 0: 环境与基线

### 目标

建立稳定的 `vLLM` 本地开发、编译、测试、评测环境。

### 任务

- 在 `WSL2 + CUDA` 中编译本地 `vLLM`
- 跑通至少一种模型的基础推理
- 跑通与 attention / quantization / cache 相关的最小测试集
- 固定一套基线模型、prompt 集、评测脚本

### 交付物

- 可复现的环境文档
- `baseline` 推理脚本
- `baseline` 性能和精度记录

### 验收标准

- 能稳定运行本地 `vLLM`
- 能跑 `Qwen2.5-7B` 或目标模型的 `greedy generation`
- 能执行最小测试子集

---

## Phase 1: FP16 等效参考算子

### 目标

建立一个高精度、可解释、可逐项对照的 `decode attention` 参考实现，作为后续量化版本的基线。

### 原则

- 先做对，再做快
- 先做可验证，再做高性能

### 任务

- 写 `PyTorch FP16/BF16` 版本 `paged decode attention`
- 输入输出与 `vLLM` 路径严格对齐
- 显式支持：
  - `query`
  - `key_cache / value_cache`
  - `block_tables`
  - `seq_lens`
  - `num_kv_heads`
  - `block_size`
- 与现有 `vLLM` kernel 做数值对比

### 重点

先把“布局对齐”做准，而不是先卷 kernel。

### 交付物

- `FP16` 参考算子实现
- 单元测试
- 对齐报告

### 目标指标

- `MaxDiff` 足够小
- `MSE` 足够低
- 能复现你后续简历里“原生等效”这条叙述

### 验收标准

- 与原始实现对齐稳定
- 输入边界条件全覆盖
- 可作为 INT8 对照基准

---

## Phase 2: KV Cache 分布分析与离线量化模拟

### 目标

搞清楚 `K/V` 的分布特性，避免盲目选量化粒度。

### 任务

- 在 `KV write` 前采集 `K/V` 张量统计
- 统计维度至少包括：
  - `per-layer`
  - `per-kv-head`
  - `per-channel`
  - `token position`
- 记录以下指标：
  - `amax`
  - `mean_abs`
  - `std`
  - `p99 / p999`
  - `outlier ratio`

### 需要横向对比的量化方案

- `per-tensor`
- `per-head`
- `per-channel`
- `per-group`

### 同时要评估的维度

- 量化误差
- scale 元数据大小
- 实现复杂度
- 与现有 `vLLM` 接口兼容性

### 关键判断

不要默认“动态 per-channel 一定最好”。

对于 `KV cache`，如果每个 token 都要携带细粒度 scale，scale 元数据本身可能抵消显存收益，甚至增加访存负担。因此更务实的路线通常是：

- `MVP`: 动态 `per-tensor` 或 `per-head`
- `最终版`: 静态 `per-channel`

### 交付物

- 分布分析报告
- 离线量化误差对比表
- 最终量化粒度决策说明

### 验收标准

- 能清楚说明为什么选择某个粒度
- 不是凭经验拍脑袋做选择

---

## Phase 3: INT8 MVP

### 目标

先实现一版能稳定运行、精度可控、改动面较小的 `INT8 KV cache MVP`。

### 推荐方案

- `INT8 symmetric quantization`
- `zero_point = 0`
- `scale = amax / 127`
- 优先支持 `per-tensor` 或 `per-head`

### 任务

- 扩展 `kv_cache_dtype=int8`
- 在 cache write 路径中加入 `INT8` 量化写入
- 在 attention 读取路径中加入 `INT8` 反量化
- 先支持一条 backend

### 这一阶段的目标不是

- 不追求最优精度
- 不追求最优性能
- 不追求全 backend 支持

### 这一阶段真正的目标

- 跑通从写入到读取的完整链路
- 建立最小可行功能闭环

### 交付物

- 第一版 `INT8 KV cache` 实现
- 单元测试
- integration test
- 与基线对比结果

### 验收标准

- 模型能完整推理
- 不出现明显错误输出或崩溃
- `logprob` 和短输出结果可控偏差

---

## Phase 4: 静态 Per-Channel 演进

### 目标

在保证显存收益的同时，提高精度并更好压制 `outliers`。

### 核心思路

如果 Phase 2 证明：

- `per-channel` 对误差改善显著
- 动态 `per-channel` 元数据开销过大

那么最终方案应转向：

- 通过离线 calibration 获得静态 `per-channel scale`
- 推理时直接使用静态 scale

### 任务

- 设计 calibration 数据收集流程
- 固定 scale 生成与加载逻辑
- 改写 `INT8` 读写路径，使之支持静态 `per-channel scale`
- 与 MVP 精度比较

### 交付物

- calibration 流程
- 静态 `per-channel` 实现
- 与 MVP 对比报告

### 验收标准

- 精度显著优于 MVP
- scale 元数据和工程复杂度可接受
- 能支撑最终端到端评测

---

## Phase 5: vLLM Attention 分支接入

### 目标

将量化算子真正接到 `vLLM` 底层 attention 路径中，形成可用的系统功能。

### 两种接入策略

#### 策略 A：最小侵入修改现有 backend

优点：

- 上手快
- 开发速度快
- 最适合第一版交付

缺点：

- 后续维护性一般
- 和 upstream 演进耦合较强

#### 策略 B：单独抽一条 custom backend

优点：

- 结构更清晰
- 更适合作为长期维护分支

缺点：

- 初期工作量更大

### 建议

先用策略 A 跑通项目，再考虑策略 B。

### 验收标准

- 可以通过配置开关切换 `baseline` / `INT8 KV cache`
- 不影响常规非量化路径

---

## Phase 6: 端到端评测

### 目标

证明该方案在真实模型推理中的工程可行性。

### 必做评测项

#### 1. 算子级

- `MaxDiff`
- `MSE`
- `logprob` close

#### 2. 系统级

- 显存占用
- `decode tokens/s`
- 平均延迟
- 长上下文下吞吐变化

#### 3. 模型级

- `HumanEval`
- `MMLU`
- `AIME24`

如果时间有限，先跑小规模验证，再跑正式全量评测。

### 推荐输出结果表

- `FP16 baseline`
- `INT8 MVP`
- `INT8 static per-channel`

每一列记录：

- 精度
- 显存
- 吞吐
- 误差指标

### 验收标准

- 精度退化可接受
- 显存收益明确
- 吞吐收益或至少没有明显负优化

---

## 5. 代码改动点清单

下面是当前本地 `vLLM` 源码中最值得关注的入口。

## 5.1 配置与 dtype

- `/mnt/f/code/vllm-code/vllm/vllm/config/cache.py`
- `/mnt/f/code/vllm-code/vllm/vllm/utils/torch_utils.py`

主要任务：

- 扩展 `CacheDType`
- 打通 `kv_cache_dtype -> torch.dtype`
- 增加 `int8` 合法性判断

## 5.2 attention backend 选择

- `/mnt/f/code/vllm-code/vllm/vllm/v1/attention/selector.py`
- `/mnt/f/code/vllm-code/vllm/vllm/v1/attention/backend.py`

主要任务：

- 让 backend 识别 `int8`
- 将当前偏 `fp8` 专用的逻辑提升为更通用的 `quantized kv cache` 判断

## 5.3 backend 接入层

- `/mnt/f/code/vllm-code/vllm/vllm/v1/attention/backends/flash_attn.py`
- `/mnt/f/code/vllm-code/vllm/vllm/v1/attention/backends/triton_attn.py`
- `/mnt/f/code/vllm-code/vllm/vllm/v1/attention/backends/flashinfer.py`

主要任务：

- 为目标 backend 声明支持 `int8`
- 接入 `reshape_and_cache_flash`
- 打通 `k_scale / v_scale`

## 5.4 C++ / CUDA 绑定层

- `/mnt/f/code/vllm-code/vllm/csrc/torch_bindings.cpp`
- `/mnt/f/code/vllm-code/vllm/csrc/cache.h`

主要任务：

- 保持 Python / C++ 接口签名一致
- 确保 `kv_cache_dtype`、`k_scale`、`v_scale` 传递稳定

## 5.5 cache 写入与读取 kernel

- `/mnt/f/code/vllm-code/vllm/csrc/cache_kernels.cu`
- `/mnt/f/code/vllm-code/vllm/csrc/cache_kernels_fused.cu`
- `/mnt/f/code/vllm-code/vllm/csrc/attention/attention_kernels.cuh`
- `/mnt/f/code/vllm-code/vllm/csrc/attention/paged_attention_v1.cu`
- `/mnt/f/code/vllm-code/vllm/csrc/attention/paged_attention_v2.cu`

主要任务：

- `INT8` 写 cache
- `INT8` 读 cache
- 量化 / 反量化 dispatch
- 尽量避免把 `INT8` 硬塞进 `FP8` 命名体系

---

## 6. 测试与验证计划

## 6.1 单元测试

优先参考：

- `/mnt/f/code/vllm-code/vllm/tests/kernels/test_cache_kernels.py`

需要补的测试：

- `INT8 reshape_and_cache`
- `INT8 round-trip`
- 不同 `block_size`
- 不同 `head_size`
- `slot_mapping` 边界

## 6.2 集成测试

优先参考：

- `/mnt/f/code/vllm-code/vllm/tests/models/quantization/test_fp8.py`

需要补的测试：

- `baseline vs int8 kv cache`
- 相同 prompt 下的 `logprob` 对比
- backend 指定情况下的功能测试

## 6.3 端到端评测

建议单独准备：

- `scripts/run_baseline_eval.sh`
- `scripts/run_int8_eval.sh`
- `scripts/collect_kv_stats.py`

---

## 7. 时间安排建议

如果按一个完整的独立项目推进，建议按 8 周节奏：

## Week 1

- 环境搭建
- `vLLM` 编译
- 基线推理与测试

## Week 2

- `FP16` 参考算子
- 对齐测试

## Week 3

- `KV` 分布采集
- 离线量化模拟
- 量化粒度决策

## Week 4

- `INT8 MVP` 写入路径
- `INT8 MVP` 读取路径
- 单元测试

## Week 5

- `vLLM` backend 接入
- integration test

## Week 6

- calibration
- 静态 `per-channel` 演进

## Week 7

- 性能优化
- 长上下文评测

## Week 8

- `HumanEval / MMLU / AIME24`
- 结果整理
- 项目总结与简历材料沉淀

---

## 8. 风险与应对

## 风险 1：动态 per-channel 元数据过大

### 应对

- Phase 2 优先做误差与元数据双评估
- MVP 不强行上动态 `per-channel`

## 风险 2：backend 改动范围过大

### 应对

- 第一版只支持一个 backend
- 优先做最小侵入实现

## 风险 3：数值误差定位困难

### 应对

- 必须先有 `FP16` 参考算子
- 每个阶段都保留独立可运行测试

## 风险 4：最终吞吐没有明显收益

### 应对

- 不只看吞吐，也看显存节省和可支持上下文长度
- 结合 `decode` 场景做分析，不只跑短序列

---

## 9. 最终交付物清单

项目完成后，至少应沉淀这些产物：

1. `FP16` 参考算子
2. `KV cache` 分布分析报告
3. `INT8 KV cache MVP`
4. 静态 `per-channel` 优化版本
5. `vLLM` 接入代码
6. 单元测试与集成测试
7. 端到端评测表
8. 项目总结文档

---

## 10. 简历表达目标

如果项目推进顺利，最终应能支撑类似下面的表达：

- 针对 `decode` 阶段 `KV cache` 访存瓶颈，完成分布分析与量化建模，确定适合工程落地的量化粒度
- 开发 `PyTorch FP16` 等效参考算子，并与原生 kernel 严格对齐，为量化改写提供高精度基线
- 在 `vLLM` attention 分支中实现 `INT8 KV cache` 路径接入，完成算子级平替
- 在 `Qwen2.5-7B` 等模型上完成端到端验证，证明显存收益、性能收益与精度保持之间的工程平衡

---

## 11. 当前最推荐的起步动作

如果现在立刻开工，我建议先做这 5 件事：

1. 跑通本地 `vLLM` 编译和最小推理
2. 读清楚 `cache.py / torch_utils.py / backend.py`
3. 写一个最小版 `FP16 paged decode attention` 参考实现
4. 补一个 `collect_kv_stats.py`
5. 先实现 `INT8 per-head MVP`

这条路线最稳，也最适合独立开发推进。

---

## 参考

- `vLLM` 本地源码：
  - `/mnt/f/code/vllm-code/vllm`
- 官方文档：
  - https://docs.vllm.ai/usage/quantization/quantized_kvcache/
  - https://docs.vllm.ai/en/latest/features/quantization/int8/
  - https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_int8/
