# `cuda-samples` 阅读建议（面向算子入门）

更新时间：2026-03-31

这份文档是根据 [operator_getting_started.md](./operator_getting_started.md) 里的学习路线，结合你当前本地的 [`cuda-samples`](../cuda-samples) 仓库整理出来的。

目标不是“把官方 sample 全看完”，而是只挑出最适合你当前阶段的内容，服务于这条主线：

1. 先建立 `kernel -> launch -> 校验 -> benchmark` 的完整直觉
2. 再建立 `shared memory / coalescing / reduction / stream / event` 的性能直觉
3. 最后再碰更高级的 Tensor Core、Graph、Driver API、图形互操作

## 先说结论

如果你现在的目标是配合 `HelloCuda` 学“算子到底在干什么”，第一轮只看下面这些 sample 就够了：

### 第一轮必须看

1. [`Samples/1_Utilities/deviceQuery`](../cuda-samples/Samples/1_Utilities/deviceQuery)
2. [`Samples/0_Introduction/vectorAdd`](../cuda-samples/Samples/0_Introduction/vectorAdd)
3. [`Samples/0_Introduction/matrixMul`](../cuda-samples/Samples/0_Introduction/matrixMul)
4. [`Samples/6_Performance/transpose`](../cuda-samples/Samples/6_Performance/transpose)
5. [`Samples/2_Concepts_and_Techniques/reduction`](../cuda-samples/Samples/2_Concepts_and_Techniques/reduction)
6. [`Samples/0_Introduction/asyncAPI`](../cuda-samples/Samples/0_Introduction/asyncAPI)
7. [`Samples/0_Introduction/simpleStreams`](../cuda-samples/Samples/0_Introduction/simpleStreams)
8. [`Samples/0_Introduction/simpleOccupancy`](../cuda-samples/Samples/0_Introduction/simpleOccupancy)

### 第二轮再看

1. [`Samples/2_Concepts_and_Techniques/scan`](../cuda-samples/Samples/2_Concepts_and_Techniques/scan)
2. [`Samples/2_Concepts_and_Techniques/shfl_scan`](../cuda-samples/Samples/2_Concepts_and_Techniques/shfl_scan)
3. [`Samples/2_Concepts_and_Techniques/threadFenceReduction`](../cuda-samples/Samples/2_Concepts_and_Techniques/threadFenceReduction)
4. [`Samples/3_CUDA_Features/globalToShmemAsyncCopy`](../cuda-samples/Samples/3_CUDA_Features/globalToShmemAsyncCopy)
5. [`Samples/4_CUDA_Libraries/matrixMulCUBLAS`](../cuda-samples/Samples/4_CUDA_Libraries/matrixMulCUBLAS)

### 现在先跳过

- `vectorAddDrv / vectorAdd_nvrtc / matrixMulDrv / matrixMul_nvrtc / matrixMulDynlinkJIT`
- `simpleGL / simpleVulkan / D3D / EGL / OpenGL` 这一类图形互操作
- `simpleMultiGPU / simpleP2P / topologyQuery / MPI` 这一类多 GPU / 分布式
- `cudaTensorCoreGemm / tf32TensorCoreGemm / bf16TensorCoreGemm / immaTensorCoreGemm`
- `simpleCudaGraphs / graphMemoryNodes / graphConditionalNodes`
- 大多数 `5_Domain_Specific` 和 `7_libNVVM`

原因很简单：这些内容不是不重要，而是它们对你当前“学算子第一性原理”的帮助，不如前面那 8 个 sample 直接。

## 为什么是这几个

`operator_getting_started.md` 里对 `cuda-samples` 的核心要求，本质上是下面几件事：

- 看懂 `grid / block / thread` 到数据元素的映射
- 看懂 `global memory` 和 `shared memory` 的区别
- 建立对 `kernel launch / synchronize / stream / event` 的基本手感
- 学会从“能跑”进入“为什么快 / 为什么慢”

本地 `cuda-samples/README.md` 当前写的是支持 `CUDA Toolkit 13.1`，样例分类也很清楚，所以完全可以按主题挑读，不需要全仓库平推。

## 推荐阅读顺序

### 0. 先确认机器和环境

先看：

- [`Samples/1_Utilities/deviceQuery`](../cuda-samples/Samples/1_Utilities/deviceQuery)

你在这里不用研究复杂实现，只需要确认：

- 你的 GPU 架构、SM 数量、shared memory、warp size、最大 threads/block
- 这些硬件参数会怎样约束后面 `matrixMul`、`reduction` 的 block 配置

这是后面读 sample 时的背景板。

### 1. 第一个必须彻底讲明白的 sample: `vectorAdd`

先看：

- [`vectorAdd/README.md`](../cuda-samples/Samples/0_Introduction/vectorAdd/README.md)
- [`vectorAdd/vectorAdd.cu`](../cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd.cu)

你在这个 sample 里要学的，不是“向量加法很简单”，而是下面这条最小闭环：

- host 侧分配内存
- `cudaMalloc`
- `cudaMemcpy`
- kernel launch
- `cudaGetLastError`
- 结果拷回 host
- correctness check

你必须能自己回答：

- 一个 thread 负责哪个元素
- 为什么 `i = blockDim.x * blockIdx.x + threadIdx.x`
- 为什么要做边界判断
- host 侧一共做了哪些 runtime API 调用

建议练习：

1. 把 `threadsPerBlock` 从 `256` 改成别的值，确认程序仍然正确
2. 自己画出 `blockIdx / threadIdx -> i` 的映射
3. 把这个 sample 改成一个最小的 `relu` 或 `axpy`

### 2. 第一个真正像“算子”的 sample: `matrixMul`

先看：

- [`matrixMul/README.md`](../cuda-samples/Samples/0_Introduction/matrixMul/README.md)
- [`matrixMul/matrixMul.cu`](../cuda-samples/Samples/0_Introduction/matrixMul/matrixMul.cu)

这是你和 `HelloCuda` 之间最重要的桥。

这个 sample 的价值不在于它是最快 GEMM，而在于它把这些东西放在一起了：

- 2D grid / 2D block
- tile-based 分块
- shared memory staging
- `__syncthreads()`
- 一次 block 计算一个输出 tile

你重点看这几个问题：

- 一个 block 负责输出矩阵的哪一块
- 一个 thread 负责输出 tile 里的哪一个元素
- `As` 和 `Bs` 为什么要放进 shared memory
- `for (k = 0; k < BLOCK_SIZE; ++k)` 这一层到底在累加什么

建议练习：

1. 把 `BLOCK_SIZE=16` 和 `BLOCK_SIZE=32` 的差异讲清楚
2. 自己画出 `A/B/C` 的 tile 访问图
3. 对照你的 `HelloCuda` 里的 GEMM，写一段“这个 sample 和工程版 GEMM 的差别”

### 3. 最适合建立访存直觉的 sample: `transpose`

注意：你学习清单里提到的 `transpose`，在当前本地仓库里实际路径是：

- [`Samples/6_Performance/transpose`](../cuda-samples/Samples/6_Performance/transpose)

先看：

- [`transpose/README.md`](../cuda-samples/Samples/6_Performance/transpose/README.md)
- [`transpose/transpose.cu`](../cuda-samples/Samples/6_Performance/transpose/transpose.cu)

这是最值得反复读的性能样例之一，因为它把“为什么同一个功能会写出完全不同性能”讲得非常直接。

你重点看这些 kernel 的演进：

- `transposeNaive`
- `transposeCoalesced`
- `transposeNoBankConflicts`
- `transposeDiagonal`

你在这里要建立的直觉是：

- coalesced access 为什么重要
- shared memory 为什么会有 bank conflict
- 为什么 `tile[TILE_DIM][TILE_DIM + 1]` 能减轻 bank conflict
- 为什么 block 调度顺序也会影响性能

如果你后面要做真正的算子优化，这个 sample 的价值通常比很多“更高级”的 sample 还高。

### 4. 最适合学 block 内协作和优化套路的 sample: `reduction`

先看：

- [`reduction/README.md`](../cuda-samples/Samples/2_Concepts_and_Techniques/reduction/README.md)
- [`reduction/reduction.cpp`](../cuda-samples/Samples/2_Concepts_and_Techniques/reduction/reduction.cpp)
- [`reduction/reduction_kernel.cu`](../cuda-samples/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu)

这个 sample 很适合你建立“性能优化是怎么一步一步长出来的”直觉。

不要只看最后一个版本，要按顺序看：

- `reduce0`
- `reduce1`
- `reduce2`
- `reduce3`
- `reduce4`

你重点看：

- 为什么最开始的写法有 divergence
- 为什么会有 shared memory bank conflict
- 为什么先在 global memory 读两份再放进 shared memory 更好
- `warp shuffle` 是在替代什么同步 / shared memory 开销

建议练习：

1. 给每个版本写一句“它比上一个版本解决了什么问题”
2. 自己总结 reduction 常见优化套路
3. 把这种优化思路迁移到你自己的简单算子里

### 5. 补 runtime 基础感觉: `asyncAPI` + `simpleStreams` + `simpleOccupancy`

先看：

- [`asyncAPI`](../cuda-samples/Samples/0_Introduction/asyncAPI)
- [`simpleStreams`](../cuda-samples/Samples/0_Introduction/simpleStreams)
- [`simpleOccupancy`](../cuda-samples/Samples/0_Introduction/simpleOccupancy)

这三个 sample 不是直接教你写算子逻辑，但它们会补齐你后面做 benchmark 和性能判断时必须有的 runtime 直觉。

#### `asyncAPI`

你要看的是：

- `cudaEventRecord`
- `cudaEventQuery`
- GPU 时间测量
- CPU 和 GPU 异步重叠

它解决的是“我怎么知道 kernel 到底跑了多久”。

#### `simpleStreams`

你要看的是：

- 为什么异步拷贝要配合 pinned memory
- 为什么多个 stream 可以让 copy 和 compute 重叠
- stream 怎么影响执行顺序

它解决的是“为什么同样的 kernel，host 侧调度方式不同，整体耗时也会差很多”。

#### `simpleOccupancy`

你要看的是：

- `cudaOccupancyMaxPotentialBlockSize`
- occupancy 和 block 配置之间的关系

它解决的是“block size 不是随便拍脑袋选的”。

## 第二轮建议看的内容

### `scan` 和 `shfl_scan`

路径：

- [`scan`](../cuda-samples/Samples/2_Concepts_and_Techniques/scan)
- [`shfl_scan`](../cuda-samples/Samples/2_Concepts_and_Techniques/shfl_scan)

这两个 sample 适合你在看完 `reduction` 之后再看。

原因：

- `scan` 比 `reduction` 更复杂
- 但它和很多高性能算子的底层组织方式有关
- 你会开始接触更强的 block 内协作模式

如果你现在刚入门，不要把它放在 `matrixMul` 前面。

### `threadFenceReduction`

路径：

- [`threadFenceReduction`](../cuda-samples/Samples/2_Concepts_and_Techniques/threadFenceReduction)

它适合在你已经理解普通 `reduction` 之后再看，用来理解：

- 单 kernel reduction
- 全局可见性
- `_threadfence()` 的意义

这已经比“算子入门第一轮”更偏同步语义了。

### `globalToShmemAsyncCopy`

路径：

- [`globalToShmemAsyncCopy`](../cuda-samples/Samples/3_CUDA_Features/globalToShmemAsyncCopy)

这是一个很好的进阶 GEMM sample，但建议放在你已经吃透 `matrixMul` 之后。

它的重点是：

- `cuda::pipeline`
- `cuda::memcpy_async`
- Ampere 及以上架构上的 global-to-shared async copy
- 多 stage pipeline

如果你还没把普通 shared memory tiling 理顺，直接看这个 sample 会比较乱。

### `matrixMulCUBLAS`

路径：

- [`matrixMulCUBLAS`](../cuda-samples/Samples/4_CUDA_Libraries/matrixMulCUBLAS)

这个 sample 不属于“手写算子原理”，但很值得作为对照项看一次。

你会知道：

- 为什么工程里很多 GEMM 不会手写
- 手写 sample 的意义主要是建立直觉
- 真正落地时往往先考虑库实现、再考虑自定义 kernel

## 哪些目录现在不值得你投入太多时间

### Driver API / NVRTC 变体

例如：

- `vectorAddDrv`
- `vectorAdd_nvrtc`
- `matrixMulDrv`
- `matrixMul_nvrtc`
- `matrixMulDynlinkJIT`

这些东西更适合在你已经掌握 runtime API 基础之后，用来补“编译和加载链路”的视角。不是当前最短路径。

### 图形互操作和平台特定样例

例如：

- `simpleGL`
- `simpleVulkan`
- `vulkanImageCUDA`
- `simpleD3D11`
- `EGLStream_*`

这些内容偏图形和平台互操作，不是算子入门主线。

### 多 GPU / 通信 / IPC

例如：

- `simpleMultiGPU`
- `simpleP2P`
- `simpleMPI`
- `streamOrderedAllocationIPC`

这些内容更偏系统和运行时，不适合第一轮。

### Tensor Core / WMMA / 更高级特性

例如：

- `cudaTensorCoreGemm`
- `tf32TensorCoreGemm`
- `bf16TensorCoreGemm`
- `immaTensorCoreGemm`

这些样例当然重要，但它们默认你已经有：

- 普通 GEMM tiling 直觉
- shared memory 直觉
- warp/block 层次的计算组织直觉

否则很容易只会“照着 API 抄”。

## 最小学习路线

如果只按你学习清单里的节奏走，我建议你在 `cuda-samples` 里这样安排：

### 第 1 批

1. `deviceQuery`
2. `vectorAdd`
3. `matrixMul`

目标：

- 彻底讲清楚 thread 映射
- 知道 shared memory 为什么存在
- 能把 GEMM tile 画出来

### 第 2 批

1. `transpose`
2. `reduction`

目标：

- 建立 coalescing 和 bank conflict 直觉
- 看懂 block 内协作
- 理解“同一个功能，优化路径可以有好几版”

### 第 3 批

1. `asyncAPI`
2. `simpleStreams`
3. `simpleOccupancy`

目标：

- 会做最基本的 timing
- 对 stream / event / overlap 有基本手感
- 不再把 block size 当成拍脑袋参数

### 第 4 批

1. `scan`
2. `threadFenceReduction`
3. `globalToShmemAsyncCopy`

目标：

- 进入更复杂的数据并行组织
- 开始接触更高级的同步与访存优化

## 每个 sample 都要输出什么

不要只是“跑过了”。

建议你每读完一个 sample，都强制自己写下这 5 个问题：

1. 一个 block 负责哪一块数据
2. 一个 thread 负责哪几个元素
3. 数据在 `global memory / shared memory / register` 之间怎么流动
4. 正确性依赖哪些同步点
5. 这个 sample 的主要性能瓶颈是什么

如果你能把这 5 个问题稳定答出来，再回头看 `HelloCuda`、Triton 或 PyTorch custom op，会顺很多。

## 对你当前阶段最有价值的源码入口

如果你只想盯住最关键的源码文件，优先顺序如下：

1. [`vectorAdd/vectorAdd.cu`](../cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd.cu)
2. [`matrixMul/matrixMul.cu`](../cuda-samples/Samples/0_Introduction/matrixMul/matrixMul.cu)
3. [`transpose/transpose.cu`](../cuda-samples/Samples/6_Performance/transpose/transpose.cu)
4. [`reduction/reduction_kernel.cu`](../cuda-samples/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu)
5. [`simpleStreams/simpleStreams.cu`](../cuda-samples/Samples/0_Introduction/simpleStreams/simpleStreams.cu)
6. [`scan/scan.cu`](../cuda-samples/Samples/2_Concepts_and_Techniques/scan/scan.cu)
7. [`globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`](../cuda-samples/Samples/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu)

## 最终建议

如果只给一个最务实的版本：

- 核心四件套：`vectorAdd + matrixMul + transpose + reduction`
- 运行时补充三件套：`asyncAPI + simpleStreams + simpleOccupancy`
- 第二轮进阶：`scan + threadFenceReduction + globalToShmemAsyncCopy`

先把这条线吃透，再去看更大的框架，收益最高。
