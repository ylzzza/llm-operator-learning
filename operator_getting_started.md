# 算子入门项目建议

更新时间：2026-03-25

这份文档面向你现在这个阶段：

- 刚开始系统学习 CUDA / GPU 算子
- 已经有 WSL2 + NVIDIA GPU 环境
- 希望做“能跑、能改、能看懂”的项目，而不是一上来就掉进大型框架

## 先说结论

如果你的目标是从零开始，尽快建立“写一个算子到底在干什么”的直觉，最适合你的顺序不是直接冲 `vLLM`、`TVM`、`TensorRT-LLM`，而是按下面这条线走：

1. 先吃透你当前这个本地项目 `HelloCuda`
2. 再做 `NVIDIA/cuda-samples`
3. 再做 `Triton tutorials`
4. 再做 `PyTorch Custom C++/CUDA Operator`
5. 最后再碰 `CUTLASS` 和更大的推理框架

原因很简单：

- `HelloCuda` 规模小，适合建立完整链路感
- `cuda-samples` 适合补 CUDA 基础动作
- `Triton` 适合快速建立 kernel 设计感
- `PyTorch custom op` 适合补“工程接入”能力
- `CUTLASS` 适合进入高性能实现，但不适合第一站

## 最适合你的项目清单

| 项目 | 推荐度 | 适合阶段 | 你会学到什么 | 为什么适合你现在 |
| --- | --- | --- | --- | --- |
| `HelloCuda`（你当前本地仓库） | 很高 | 第一站 | kernel、bench、Python 绑定、测试 | 体量可控，链路完整 |
| `NVIDIA/cuda-samples` | 很高 | 第一站到第二站 | CUDA 基础 API、内存、线程组织、经典样例 | 官方、碎片小、容易验证 |
| `Triton tutorials` | 很高 | 第二站 | 向量加法、softmax、matmul、fused kernel | 上手快，反馈快，适合理解算子思维 |
| `PyTorch Custom C++/CUDA Operators` | 很高 | 第二站到第三站 | 自定义算子注册、调度、测试、与 PyTorch 集成 | 很接近真实工作流 |
| `NVIDIA/CUTLASS` | 中高 | 第三站以后 | 高性能 GEMM、分块、层次化并行、模板抽象 | 很重要，但学习曲线明显更陡 |
| `tinygrad` | 中 | 第三站以后 | 图、IR、kernel lowering、轻量框架结构 | 适合建立“框架如何看算子”的视角 |

## 逐个怎么学

### 1. `HelloCuda`

这是你现在最该优先啃的项目。

你当前仓库的 [README.md](/mnt/f/code/operator-code/HelloCuda/README.md) 已经明确把目标收敛在：

- Linux/WSL
- CUDA 算子库
- 当前重点是 GEMM
- 同时覆盖 C++/CUDA、bench CLI、PyTorch extension、Python UT

这很适合入门，因为它不是“只会写一个 `.cu` 文件”的 demo，而是一个小型但完整的算子工程。

建议你按这个顺序读和改：

1. 先跑通构建和测试
2. 看 `src/models/rtx3060ti/gemm/kernels/` 下几个 kernel 的差别
3. 理解 `dispatch`、`wrapper`、`register` 这一层怎么把实现组织起来
4. 再看 `python_ext/` 怎么把算子接到 Python
5. 最后用 `bench` 和测试去验证你的修改

建议你在这个项目里做的第一个练习：

- 自己把 `gpu_naive` 讲明白
- 再把 `tiled64`、`tiled128` 的分块策略画出来
- 给 `bench` 增加一个你自己的 kernel 开关
- 做一次性能对比和正确性验证

如果你能把这个项目里的 GEMM 路线吃透，你已经不算“从零开始”了。

### 2. `NVIDIA/cuda-samples`

项目地址：

- https://github.com/NVIDIA/cuda-samples

为什么值得做：

- 这是 NVIDIA 官方样例库
- README 明确把样例分成 `Introduction`、`Concepts and Techniques`、`Performance`、`CUDA Libraries` 等类别
- 当前 GitHub 版本说明支持 `CUDA Toolkit 13.0`

适合你的学习方式不是“全做”，而是只挑最有价值的样例：

- `vectorAdd`
- `matrixMul`
- `transpose`
- `reduction`
- 跟 memory copy、stream、event 相关的基础样例

你在这里主要补的是：

- grid / block / thread 的映射
- global memory 访问模式
- shared memory 的基本使用
- kernel launch 和同步
- CUDA runtime API 的基本手感

一句话判断：如果你现在看到一个 kernel 还不能立刻说出“每个线程算哪一块数据”，就先多做 `cuda-samples`。

### 3. `Triton tutorials`

项目与文档：

- https://triton-lang.org/main/getting-started/tutorials/
- https://github.com/triton-lang/triton

为什么值得做：

- Triton 官方教程明确建议按顺序阅读教程
- 教程覆盖 `Vector Addition`、`Fused Softmax`、`Matrix Multiplication`、`Layer Normalization`、`Fused Attention`
- 它很适合训练“一个算子应该怎样切块、怎样融合、怎样减少访存”的直觉

它对你最有价值的地方在于：

- 比 CUDA C++ 更快看到结果
- 比直接读大型框架更容易形成算子思维
- 非常适合把“算子”看成数据排布和并行映射问题

建议你在 Triton 里至少做完这三个：

1. `Vector Addition`
2. `Matrix Multiplication`
3. `Fused Softmax`

做完后你应该能回答这些问题：

- 为什么要分块
- 为什么 block size 会影响性能
- 为什么融合算子能减少内存流量
- 为什么同一个功能可以写出完全不同的性能结果

### 4. `PyTorch Custom C++/CUDA Operators`

官方教程：

- https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html

为什么值得做：

- 这是 PyTorch 官方推荐的自定义 C++/CUDA operator 路径
- 教程重点不只是“写一个 kernel”，还包括注册、打包、测试和与 PyTorch 子系统协作
- 文档当前版本说明最后更新时间是 2026-01-20

这一步很关键，因为真实工作里你写算子通常不是裸写 `.cu` 就结束，而是要解决：

- Python 侧怎么调
- shape / dtype / device 怎么校验
- 前向和反向怎么接
- 单测怎么写
- 怎么让工程可复现地构建

建议你在这一步做一个非常小的自定义算子：

- 先做 `elementwise add`
- 再做 `relu + bias` 融合
- 最后再尝试一个简化版 `matmul`

如果你能独立写出一个 PyTorch custom op，并且自己测通，你就已经跨过“会写 kernel”和“会做算子工程”之间的门槛了。

### 5. `NVIDIA/CUTLASS`

项目地址：

- https://github.com/NVIDIA/cutlass

为什么重要：

- CUTLASS 是 NVIDIA 的高性能线性代数模板库
- GitHub 首页当前显示版本为 `CUTLASS 4.3.5`
- 它非常适合学习高性能 GEMM 是如何分层拆解的

但为什么不建议你第一站就学：

- 模板和抽象层很多
- 一上来容易只会“抄配置”，不知道真正发生了什么
- 如果没有自己的朴素 GEMM 和 tiled GEMM 经验，读起来会比较痛苦

你应该在什么时候学 CUTLASS：

- 当你已经能自己写 naive GEMM
- 知道 shared memory tiling 的意义
- 能看懂 warp/block 级别的数据搬运

到那时再看 CUTLASS，会非常有收获；在那之前，它更像“高级答案”，不是“入门教材”。

### 6. `tinygrad`

项目地址：

- https://github.com/tinygrad/tinygrad

为什么可以作为补充：

- tinygrad 的目标就是保持系统足够小、足够可读
- README 明确提到它包含 tensor library、autograd、IR/compiler、JIT 和 graph execution
- 它可以帮助你理解：框架层是怎样把上层操作逐步 lower 到底层 kernel 的

它适合你的时机：

- 你已经不满足于“只会写一个 kernel”
- 你想开始理解图、调度、lowering、fusion 这些词到底是什么意思

但它不应该是你的第一站，因为它更像“轻量框架学习”，不是“单算子工程入门”。

## 我给你的推荐顺序

### 第一阶段：把算子写出来

目标：理解 kernel、线程组织、shared memory、正确性校验

顺序：

1. `HelloCuda`
2. `cuda-samples`

完成标准：

- 你能自己解释一个 GEMM kernel 的线程映射
- 你能改一个 tile size 并验证结果
- 你知道什么时候需要 shared memory

### 第二阶段：把算子接进框架

目标：理解算子不仅是 kernel，还是接口、调度、测试、构建

顺序：

1. `Triton tutorials`
2. `PyTorch custom op`

完成标准：

- 你能独立做一个简单 fused op
- 你会写最基本的 benchmark 和 UT
- 你知道 Python 调到 CUDA kernel 中间经过了哪些层

### 第三阶段：开始看高性能和真实框架

目标：理解工业级高性能实现和大型推理框架里的算子位置

顺序：

1. `CUTLASS`
2. `tinygrad`
3. 再看 `vLLM`、`TensorRT-LLM`、`TVM`

完成标准：

- 你能大致看懂高性能 GEMM 框架的目录结构
- 你知道框架是在什么位置插入自定义 kernel 的
- 你能判断一个项目是在做算子、调度、编译，还是运行时

## 哪些项目不建议你一上来就学

### `vLLM`

不建议作为第一站。

原因：

- 系统目标是推理服务，不是算子教学
- 代码量大，模块边界多
- 你会很容易把精力花在调度、cache、runtime、distributed 上，而不是算子本身

### `TVM`

也不建议作为第一站。

原因：

- 编译器和调度抽象很多
- 学到的是“如何生成/调度算子”，不是最直接的“手写算子第一性原理”

### `TensorRT-LLM`

更不建议作为第一站。

原因：

- 工程体量更大
- 推理优化链路更复杂
- 对基础 CUDA / kernel 直觉要求更高

## 你现在最现实的学习方案

如果只考虑“最适合你上手”，我建议你未来 4 到 6 周这样安排：

### 第 1 周

- 配好 WSL CUDA 开发环境
- 跑通 `HelloCuda`
- 理解 naive GEMM

### 第 2 周

- 改 `HelloCuda` 的 tiled kernel
- 做 benchmark
- 写下你自己的性能观察

### 第 3 周

- 做 `cuda-samples` 的 `vectorAdd / matrixMul / transpose / reduction`
- 把每个 sample 的线程映射画出来

### 第 4 周

- 做 `Triton` 的 `Vector Add / Matmul / Softmax`
- 对比 Triton 和 CUDA C++ 在表达上的区别

### 第 5 到 6 周

- 做一个 PyTorch custom C++/CUDA op
- 给它补测试
- 再回头优化 `HelloCuda`

## 最终建议

如果只让我给你一个最务实的答案：

- 第一优先级：继续做你手上的 `HelloCuda`
- 第二优先级：补 `cuda-samples`
- 第三优先级：做 `Triton tutorials`
- 第四优先级：做 `PyTorch custom op`
- 暂时不要把主精力放在 `vLLM`、`TVM`、`TensorRT-LLM`

你现在最需要建立的是：

- 一个 kernel 是怎么映射线程的
- 一个算子工程是怎么组织代码、构建、测试和 benchmark 的
- 一个算子怎么从 Python 一路调到 CUDA 实现

这三件事情打通之后，再去看大框架，你会快很多。

## 参考链接

- `HelloCuda` 本地说明：[README.md](/mnt/f/code/operator-code/HelloCuda/README.md)
- NVIDIA CUDA Samples: https://github.com/NVIDIA/cuda-samples
- Triton Tutorials: https://triton-lang.org/main/getting-started/tutorials/
- Triton GitHub: https://github.com/triton-lang/triton
- PyTorch Custom C++/CUDA Operators: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- CUTLASS: https://github.com/NVIDIA/cutlass
- tinygrad: https://github.com/tinygrad/tinygrad
