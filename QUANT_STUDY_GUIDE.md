# CANN `ops-nn/quant` 量化算子学习手册

这份文档不是在罗列目录，而是帮你回答两个问题：

1. 我应该先看哪些算子。
2. 我看一个量化算子时，应该按什么顺序看。

如果你的目标是“学会这套量化算子的实现套路”，建议先读 `group_quant`，再读 `trans_quant_param_v2`，然后读 `dynamic_quant_v2` 和 `ascend_quant_v2`。

## 1. 先建立整体认知

`ops-nn/quant/CMakeLists.txt` 的作用很简单：遍历每个子目录，把带 `CMakeLists.txt` 的算子目录加进构建。也就是说，`ops-nn/quant` 本质上就是“量化相关算子集合”。

当前目录下的子算子大致可以分成 4 类：

| 类别 | 目录 | 适合什么时候看 |
| --- | --- | --- |
| 直接量化 | `quantize`、`ascend_quant`、`ascend_quant_v2`、`group_quant` | 想先理解“给定 scale/offset 后怎么量化”时看 |
| 动态量化 | `dynamic_quant`、`dynamic_quant_v2`、`dynamic_block_quant`、`dynamic_mx_quant`、`dynamic_mx_quant_with_dual_axis`、`dynamic_dual_level_mx_quant`、`grouped_dynamic_block_quant`、`grouped_dynamic_mx_quant` | 想理解“scale 运行时怎么计算、怎么算 tiling、怎么分支”时看 |
| 融合类/场景类 | `dynamic_quant_update_scatter`、`dynamic_quant_update_scatter_v2`、`swi_glu_quant`、`dequant_swiglu_quant`、`flat_quant` | 前面基础看懂以后再看 |
| 反量化/参数/训练辅助 | `dequant_bias`、`trans_quant_param`、`trans_quant_param_v2`、`fake_quant_affine_cachemask`、`ifmr`、`ascend_anti_quant_v2` | 想补齐量化链路上下游时看 |

补充两点：

- `ascend_anti_quant_v2` 当前公开目录里主要是接口和样例，不是完整 AscendC 实现入口。
- 带 `v2` 的算子通常更值得优先看，因为功能更完整、约束更新、目录也更接近当前主线。

## 2. 一个量化算子目录里该先看什么

大部分量化算子目录都有下面这些层：

| 目录/文件 | 作用 | 学的时候重点看什么 |
| --- | --- | --- |
| `README.md` | 算子语义说明 | 公式、输入输出、dtype、shape 约束、支持芯片 |
| `docs/aclnn*.md` | ACLNN 接口文档 | API 形参、调用方式、约束 |
| `examples/test_aclnn_*.cpp` | 最小调用样例 | 输入怎么构造、两段式接口怎么调用 |
| `op_host/*_def.cpp` | 算子定义注册 | 输入输出个数、属性、dtype 组合、芯片配置 |
| `op_host/*_infershape.cpp` | shape/dtype 推导 | 输出 shape、输出 dtype、属性影响 |
| `op_host/*_tiling.cpp` | host 侧调度与检查 | 参数校验、blockDim、workspace、tiling key |
| `op_kernel/*.cpp` / `*.h` | AICore kernel | kernel 入口、不同分支、真正计算逻辑 |
| `op_graph/*_proto.h` | 图模式 IR | 构图接口长什么样 |
| `op_api/*.cpp` | API 封装层 | ACLNN/l0op 如何把参数组织到 launcher |
| `framework/*plugin*.cpp` | 插件对接层 | ONNX/plugin 桥接怎么做 |
| `tests/ut/op_host` | host UT | shape/tiling/infershape 是否符合预期 |
| `tests/ut/op_kernel` | kernel UT | kernel 分支是否能跑通 |
| `tests/st` | 系统/端到端测试 | 真实接口路径怎么验 |

你看任意一个算子，都建议固定按下面顺序走：

1. 先看 `README.md`，把公式和约束搞清楚。
2. 再看 `examples/test_aclnn_*.cpp`，确认调用链路。
3. 看 `op_host/*_def.cpp`，确认算子 contract。
4. 看 `op_host/*_infershape.cpp`，确认输出是怎么推出来的。
5. 看 `op_host/*_tiling.cpp`，确认 host 侧怎么选 kernel 分支。
6. 看 `op_kernel/*.cpp`，找到 kernel 入口和 `TILING_KEY_IS(...)` 分支。
7. 最后看 `tests`，确认作者到底在验证什么。

## 3. 推荐入门路径

### 路线 A：想学“怎么写一个量化算子”

这是最推荐的路线。

#### 第一站：`group_quant`

这是最适合入门的完整样板之一。原因：

- 语义不复杂：输入 `x`、`scale`、`group_index`、可选 `offset`，输出 `y`。
- host 侧定义、infershape、tiling、kernel、UT 都很完整。
- kernel 入口不长，容易看懂 host 和 kernel 的衔接关系。

建议阅读顺序：

- [group_quant/README.md](group_quant/README.md)
- [group_quant/examples/test_aclnn_group_quant.cpp](group_quant/examples/test_aclnn_group_quant.cpp)
- [group_quant/op_host/group_quant_def.cpp](group_quant/op_host/group_quant_def.cpp)
- [group_quant/op_host/group_quant_infershape.cpp](group_quant/op_host/group_quant_infershape.cpp)
- [group_quant/op_host/group_quant_tiling.cpp](group_quant/op_host/group_quant_tiling.cpp)
- [group_quant/op_kernel/group_quant.cpp](group_quant/op_kernel/group_quant.cpp)
- [group_quant/op_kernel/group_quant_base.h](group_quant/op_kernel/group_quant_base.h)
- [group_quant/tests/ut/op_host/test_group_quant_tiling.cpp](group_quant/tests/ut/op_host/test_group_quant_tiling.cpp)
- [group_quant/tests/ut/op_kernel/test_group_quant.cpp](group_quant/tests/ut/op_kernel/test_group_quant.cpp)

这一站你要重点学会：

- `OpDef` 是怎么描述输入输出和属性的。
- `InferShape/InferDataType` 是怎么写的。
- `TilingContext` 里会做哪些检查。
- `TILING_KEY_IS(...)` 怎么把 host 分支和 kernel 分支接起来。

#### 第二站：`trans_quant_param_v2`

这个算子非常适合拿来理解“一个更小、更干净的量化辅助算子”。

它的特点是：

- 不是对激活值做量化，而是把 `scale/offset` 转成硬件需要的打包格式。
- kernel 逻辑集中，tiling 逻辑也比较短。
- 很适合补齐“量化参数最终是怎么喂给硬件/下游算子的”这块认知。

建议阅读顺序：

- [trans_quant_param_v2/README.md](trans_quant_param_v2/README.md)
- [trans_quant_param_v2/examples/test_aclnn_trans_quant_param_v2.cpp](trans_quant_param_v2/examples/test_aclnn_trans_quant_param_v2.cpp)
- [trans_quant_param_v2/op_host/trans_quant_param_v2_infershape.cpp](trans_quant_param_v2/op_host/trans_quant_param_v2_infershape.cpp)
- [trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.cpp](trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.cpp)
- [trans_quant_param_v2/op_kernel/trans_quant_param_v2.cpp](trans_quant_param_v2/op_kernel/trans_quant_param_v2.cpp)
- [trans_quant_param_v2/op_kernel/trans_quant_param_v2.h](trans_quant_param_v2/op_kernel/trans_quant_param_v2.h)

这一站重点不是“量化公式本身”，而是：

- 参数打包。
- workspace/tiling 的最小闭环。
- 一个非大算子的 host-kernel 配合方式。

#### 第三站：`dynamic_quant_v2`

看完前两个，再看这个会顺很多。

它能帮你理解：

- 动态量化怎么在运行时算 `scale/offset`。
- 可选输入 `smooth_scales`、`group_index` 怎么进入分支逻辑。
- 一个算子怎么支持多种量化模式和多种输出类型。
- 同一个 kernel 入口里怎么根据 tiling key 调多个实现。

建议阅读顺序：

- [dynamic_quant_v2/README.md](dynamic_quant_v2/README.md)
- [dynamic_quant_v2/examples/test_aclnn_dynamic_quant_v2.cpp](dynamic_quant_v2/examples/test_aclnn_dynamic_quant_v2.cpp)
- [dynamic_quant_v2/op_host/dynamic_quant_v2_def.cpp](dynamic_quant_v2/op_host/dynamic_quant_v2_def.cpp)
- [dynamic_quant_v2/op_host/dynamic_quant_v2_infershape.cpp](dynamic_quant_v2/op_host/dynamic_quant_v2_infershape.cpp)
- [dynamic_quant_v2/op_kernel/dynamic_quant_v2.cpp](dynamic_quant_v2/op_kernel/dynamic_quant_v2.cpp)
- [dynamic_quant/op_host/dynamic_quant_tiling.cpp](dynamic_quant/op_host/dynamic_quant_tiling.cpp)
- [dynamic_quant/op_kernel/dynamic_quant.h](dynamic_quant/op_kernel/dynamic_quant.h)
- [dynamic_quant/op_kernel/dynamic_quant_db.h](dynamic_quant/op_kernel/dynamic_quant_db.h)
- [dynamic_quant/op_kernel/dynamic_quant_moe.h](dynamic_quant/op_kernel/dynamic_quant_moe.h)

为什么这里要连 `dynamic_quant` 一起看：

- `dynamic_quant_v2/op_kernel/dynamic_quant_v2.cpp` 直接复用了 `dynamic_quant` 目录下的大量 kernel 实现。
- 所以 `v2` 更多是在接口、能力和分支上扩展，真正的核心实现有一部分还在 `dynamic_quant` 里。

#### 第四站：`ascend_quant_v2`

这是更接近“生产级静态量化主线”的实现。

它能帮你理解：

- `axis`、`round_mode`、`dst_type` 这些属性怎样影响 host 和 kernel。
- per-tensor / per-channel / per-head 是怎么分支的。
- 不同芯片和不同 dtype 组合，host 侧要怎么做合法性校验。
- 一个相对成熟的算子是怎么拆成多个 kernel 头文件和 tiling 文件的。

建议阅读顺序：

- [ascend_quant_v2/README.md](ascend_quant_v2/README.md)
- [ascend_quant_v2/examples/test_aclnn_ascend_quant.cpp](ascend_quant_v2/examples/test_aclnn_ascend_quant.cpp)
- [ascend_quant_v2/op_host/ascend_quant_v2_def.cpp](ascend_quant_v2/op_host/ascend_quant_v2_def.cpp)
- [ascend_quant_v2/op_host/ascend_quant_v2_infershape.cpp](ascend_quant_v2/op_host/ascend_quant_v2_infershape.cpp)
- [ascend_quant_v2/op_host/ascend_quant_v2_tiling.cpp](ascend_quant_v2/op_host/ascend_quant_v2_tiling.cpp)
- [ascend_quant_v2/op_kernel/ascend_quant_v2.cpp](ascend_quant_v2/op_kernel/ascend_quant_v2.cpp)
- [ascend_quant_v2/op_kernel/ascend_quant_v2_fp16.h](ascend_quant_v2/op_kernel/ascend_quant_v2_fp16.h)
- [ascend_quant_v2/op_kernel/ascend_quant_v2_fp32.h](ascend_quant_v2/op_kernel/ascend_quant_v2_fp32.h)

### 路线 B：想先快速理解“接口到实现”的全流程

如果你暂时不想一上来就啃复杂 tiling，可以按这个顺序：

1. `quantize`
2. `group_quant`
3. `dynamic_quant_v2`

对应建议文件：

- [quantize/README.md](quantize/README.md)
- [quantize/examples/test_aclnn_quantize.cpp](quantize/examples/test_aclnn_quantize.cpp)
- [quantize/op_api/quantize.cpp](quantize/op_api/quantize.cpp)
- [quantize/op_host/quantize_infershape.cpp](quantize/op_host/quantize_infershape.cpp)
- [quantize/op_kernel/quantize_apt.cpp](quantize/op_kernel/quantize_apt.cpp)

这条路线更适合先建立“API 调用 + dtype 映射 + kernel 入口”概念，但如果你的目标是写算子实现，还是建议尽快切回路线 A。

## 4. 各目录更适合解决什么问题

| 目录 | 建议优先级 | 你主要能学到什么 |
| --- | --- | --- |
| `group_quant` | 很高 | 最完整、最适合入门的 host+kernel 样板 |
| `trans_quant_param_v2` | 很高 | 量化参数打包、短链路 tiling、硬件参数视角 |
| `dynamic_quant_v2` | 很高 | 动态量化、多输出、可选输入、复杂分支 |
| `ascend_quant_v2` | 很高 | 生产感更强的静态量化实现 |
| `quantize` | 高 | ACLNN/API 视角下的标准量化调用链 |
| `dynamic_quant` | 高 | `dynamic_quant_v2` 的实现底座之一 |
| `fake_quant_affine_cachemask` | 中 | 训练侧 fake quant、多输出 `y + mask` |
| `ifmr` | 中 | 量化校准/搜索最优 scale、offset 的思路 |
| `dequant_bias` | 中 | 反量化基本形态 |
| `dynamic_block_quant` | 中 | block 级动态量化 |
| `dynamic_mx_quant` | 中 | MX 量化基本模式 |
| `grouped_dynamic_mx_quant` | 中 | 分组 + MX 量化 |
| `dynamic_mx_quant_with_dual_axis` | 低 | 双轴 MX，适合后看 |
| `dynamic_dual_level_mx_quant` | 低 | 双层 MX，适合后看 |
| `grouped_dynamic_block_quant` | 低 | grouped block 量化 |
| `swi_glu_quant` | 低 | 激活融合量化 |
| `dequant_swiglu_quant` | 低 | dequant + swiglu + quant 融合 |
| `flat_quant` | 低 | 融合算子，适合基础扎实后再看 |
| `dynamic_quant_update_scatter*` | 低 | 融合路径优化，不适合入门第一站 |
| `ascend_anti_quant_v2` | 低 | 当前公开目录更偏接口和样例，不适合作为实现入口 |

## 5. 不建议一开始就看的目录

下面这些我建议你放后面：

- `dynamic_mx_quant_with_dual_axis`
- `dynamic_dual_level_mx_quant`
- `flat_quant`
- `swi_glu_quant`
- `dequant_swiglu_quant`
- `dynamic_quant_update_scatter`
- `dynamic_quant_update_scatter_v2`

原因很简单：这些目录不是“不会写”，而是融合、场景、分支和约束都更多。你如果还没把 `def / infershape / tiling / kernel entry / tests` 这条主线看顺，很容易陷在业务细节里。

## 6. 看一个算子时你要盯住的几个问题

每看一个目录，都建议你带着下面这些问题读：

1. 这个算子的数学定义是什么，属于静态量化还是动态量化？
2. `scale`、`offset`、`group_index`、`smooth_scales` 这些量是输入给的，还是运行时算出来的？
3. 输出 shape/dtype 是直接继承输入，还是受属性影响？
4. host 侧在哪做参数合法性校验？
5. tiling key 有几个，每个 key 对应哪条 kernel 分支？
6. kernel 主体是写在 `*.cpp`，还是被拆到 `*.h` 模板里？
7. tests 重点在测什么，是 infershape、tiling，还是 kernel 行为？

只要你每次都把这 7 个问题答出来，这个目录基本就算看明白了。

## 7. 一周入门建议

如果你想按节奏推进，可以这么安排：

### 第 1 天

- 看 [CMakeLists.txt](CMakeLists.txt)
- 看 [group_quant/README.md](group_quant/README.md)
- 看 [group_quant/examples/test_aclnn_group_quant.cpp](group_quant/examples/test_aclnn_group_quant.cpp)

目标：先把“一个量化算子对外怎么被调用”看明白。

### 第 2 天

- 看 [group_quant/op_host/group_quant_def.cpp](group_quant/op_host/group_quant_def.cpp)
- 看 [group_quant/op_host/group_quant_infershape.cpp](group_quant/op_host/group_quant_infershape.cpp)
- 看 [group_quant/op_host/group_quant_tiling.cpp](group_quant/op_host/group_quant_tiling.cpp)

目标：把 host 侧三件事看明白，定义、推导、调度。

### 第 3 天

- 看 [group_quant/op_kernel/group_quant.cpp](group_quant/op_kernel/group_quant.cpp)
- 看 [group_quant/op_kernel/group_quant_base.h](group_quant/op_kernel/group_quant_base.h)
- 看 [group_quant/tests/ut/op_host/test_group_quant_tiling.cpp](group_quant/tests/ut/op_host/test_group_quant_tiling.cpp)

目标：把 tiling 和 kernel 怎么接上看明白。

### 第 4 天

- 看 [trans_quant_param_v2/README.md](trans_quant_param_v2/README.md)
- 看 [trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.cpp](trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.cpp)
- 看 [trans_quant_param_v2/op_kernel/trans_quant_param_v2.cpp](trans_quant_param_v2/op_kernel/trans_quant_param_v2.cpp)

目标：补齐量化参数在硬件侧的视角。

### 第 5 天

- 看 [dynamic_quant_v2/README.md](dynamic_quant_v2/README.md)
- 看 [dynamic_quant_v2/op_host/dynamic_quant_v2_def.cpp](dynamic_quant_v2/op_host/dynamic_quant_v2_def.cpp)
- 看 [dynamic_quant_v2/op_host/dynamic_quant_v2_infershape.cpp](dynamic_quant_v2/op_host/dynamic_quant_v2_infershape.cpp)

目标：理解多输出、动态量化、可选输入。

### 第 6 天

- 看 [dynamic_quant_v2/op_kernel/dynamic_quant_v2.cpp](dynamic_quant_v2/op_kernel/dynamic_quant_v2.cpp)
- 看 [dynamic_quant/op_host/dynamic_quant_tiling.cpp](dynamic_quant/op_host/dynamic_quant_tiling.cpp)
- 看 [dynamic_quant/op_kernel/dynamic_quant.h](dynamic_quant/op_kernel/dynamic_quant.h)

目标：开始理解复杂 kernel 分支。

### 第 7 天

- 看 [ascend_quant_v2/README.md](ascend_quant_v2/README.md)
- 看 [ascend_quant_v2/op_host/ascend_quant_v2_tiling.cpp](ascend_quant_v2/op_host/ascend_quant_v2_tiling.cpp)
- 看 [ascend_quant_v2/op_kernel/ascend_quant_v2.cpp](ascend_quant_v2/op_kernel/ascend_quant_v2.cpp)

目标：把“更真实的主线量化算子”补上。

## 8. 你可以直接用的几个命令

先看全量目录：

```bash
find ops-nn/quant -maxdepth 1 -type d | sort
```

看某个算子的完整文件分布：

```bash
find ops-nn/quant/group_quant -maxdepth 2 -type f | sort
```

快速找一个算子的 host/kernel 关键入口：

```bash
rg -n "IMPL_OP_INFERSHAPE|IMPL_OP_OPTILING|__global__ __aicore__|OP_ADD" ops-nn/quant/group_quant
```

快速扫所有量化 README：

```bash
for f in ops-nn/quant/*/README.md; do
  echo "### $f"
  rg -n "功能说明|算子功能|计算公式|调用说明" "$f"
done
```

## 9. 最后给你的结论

如果你现在时间有限，不要平均用力。最值得你先吃透的是这 4 个目录：

1. `group_quant`
2. `trans_quant_param_v2`
3. `dynamic_quant_v2`
4. `ascend_quant_v2`

如果这 4 个你已经能自己顺着 `README -> example -> def -> infershape -> tiling -> kernel -> tests` 讲清楚，那么你再去看其他量化算子，基本都只是“场景变化”和“细节扩展”，不会再觉得目录很散。
