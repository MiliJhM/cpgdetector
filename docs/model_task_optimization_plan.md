# CpGDetector Model and Task Optimization Plan

本文档整理当前 CpGDetector 主模型在训练性能、模型设计和任务设计上的优化方案。目标不是单纯提高单步速度，而是在不削弱 CpG island 逐碱基分割能力的前提下，提高 GPU 利用率、降低 CPU/DataLoader 压力，并减少 base/window 多任务之间的冲突。

## 1. 当前设计判断

当前主模型是一个共享 1D CNN encoder，加两个任务头：

- base head：逐碱基 CpG island segmentation。
- window head：预测窗口内是否存在 CpG island 或窗口强度。
- 多任务损失：base BCE/Dice、window loss、base-window consistency，并支持 fixed、uncertainty、GradNorm。

这个总体方向是合理的。CpG island 是区域性序列特征，逐碱基分割是比窗口分类更贴近最终 interval prediction 的任务形式；window head 作为辅助任务也有价值，可以帮助 encoder 学到窗口级别的 CpG 富集信号。

主要问题在于：

- 输入管线仍然会产生较大的 CPU batch 组装和 one-hot 开销。
- GradNorm 每 step 额外计算梯度范数，训练成本高，并且会禁用 `torch.compile`。
- 当前 encoder 全程保持 full resolution，所有通道都在 `window_size` 长度上计算，FLOPs 随 batch 和 channel 放大明显。
- window binary target 使用 `fraction > 0`，边界窗口和极低 overlap 窗口会带来标签噪声。
- 验证任务是逐碱基统计，大验证集每 epoch 代价很高。

## 2. 优化优先级概览

建议按以下顺序推进：

| 阶段 | 方向 | 预期收益 | 风险 |
| --- | --- | --- | --- |
| P0 | 输入和训练循环减负 | 直接改善吞吐，低风险 | 需要小改模型输入 |
| P1 | 调整多任务训练策略 | 恢复 compile、减少反向开销 | 需要重新比较收敛曲线 |
| P2 | window 任务目标重构 | 降低任务冲突，提高泛化 | 需要系统 ablation |
| P3 | 模型结构轻量化/多尺度化 | 降低 FLOPs，提高大 batch 能力 | 需要更多实验验证 |
| P4 | 边界/距离辅助任务 | 改善 interval 边界质量 | 增加任务设计复杂度 |

## 3. P0: 输入管线和训练循环优化

### 3.1 将 one-hot 从 CPU collate 移到 GPU 或模型内

当前 DataLoader 返回 `seq_idx`，collate 阶段把它转成 `(B, 4, L)` float one-hot。对于 `batch_size=4096, window_size=512`，CPU 每个 batch 需要构造约 4096 x 4 x 512 个 float，且需要 pin/copy 到 GPU。

建议改为：

- DataLoader/collate 只返回 `(B, L)` 的 `seq_idx`。
- 在训练循环中将 `seq_idx` 传到 GPU。
- 模型内部二选一：
  - `F.one_hot(seq_idx, 4).permute(...).float()` 在 GPU 上生成 one-hot。
  - 使用 `nn.Embedding(4, embed_dim)`，再转成 Conv1d 输入。

首选实验：

- A 版：GPU one-hot，保持原模型第一层 `in_channels=4` 不变，行为最接近当前模型。
- B 版：`nn.Embedding(4, 8 或 16)`，第一层 Conv 输入通道变为 embed_dim，可能提升表达力，但会改变模型容量。

评估指标：

- dataloader wait time。
- step time。
- GPU utilization。
- peak CPU memory。
- val base PR-AUC/F1 是否保持。

### 3.2 使用 bf16/TF32 优先于 fp16

A100 上建议默认使用 bf16 autocast，而不是 fp16。bf16 范围更大，对 BCE、Dice、GradNorm 这类混合损失更稳。

建议配置项：

- `training.amp_dtype: bf16`
- CUDA 上启用 TF32：
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`

验证：

- 与 fp16 比较 train loss 是否出现 NaN/Inf。
- 比较 step time。
- 比较最终 validation metrics。

### 3.3 评估降频和分层验证

逐碱基验证非常贵。建议拆成两类：

- fast validation：固定抽样 5k-20k windows，每 epoch 跑，用于 early stopping 和 LR scheduler。
- full validation：每 5 个 epoch 或 best checkpoint 时跑一次，用于报告和最终选择。

这样能避免训练大部分时间耗在验证集上。

建议新增配置：

```yaml
evaluation:
  fast_val_windows: 20000
  full_eval_interval: 5
  full_eval_on_best: true
```

## 4. P1: 多任务学习策略优化

### 4.1 GradNorm 不应作为默认大规模训练策略

GradNorm 的优势是动态平衡任务，但代价是每 step 对共享参数额外调用 `autograd.grad`。当前实现还会禁用 `torch.compile`，这在 A100 上代价明显。

建议默认策略：

- full training 默认使用 `uncertainty` 或 `fixed`。
- GradNorm 只作为实验项，或低频更新。

可选实现：

1. **低频 GradNorm**

   每 `gradnorm_update_interval` step 更新一次任务权重，其余 step 使用上一次权重。

   ```yaml
   training:
     mtl_method: gradnorm
     gradnorm_update_interval: 10
   ```

2. **Warmup GradNorm**

   前 N epoch 使用 GradNorm 找到合理权重，之后冻结权重并恢复 compile。

   ```yaml
   training:
     gradnorm_warmup_epochs: 3
     freeze_mtl_weights_after_warmup: true
   ```

3. **Shared tail GradNorm**

   只在 encoder 最后一层或 task adapter 输入特征上计算梯度范数，而不是全共享 encoder 参数。

推荐实验顺序：

- fixed weights。
- uncertainty。
- low-frequency GradNorm。
- full GradNorm 作为对照。

### 4.2 consistency loss 分阶段打开

base-window consistency 对任务对齐有帮助，但训练早期 base head 还不稳定，过早强约束可能让两个 head 互相拖累。

建议：

- 前 1-3 epoch 关闭 consistency。
- 随 epoch 线性 warmup 到目标权重。

```yaml
training:
  lambda_consistency: 0.10
  consistency_warmup_epochs: 3
```

验证：

- early epoch base recall 是否提升。
- window PR-AUC 是否更平滑。
- base/window loss 是否减少震荡。

## 5. P2: Window 任务目标重构

### 5.1 避免 `fraction > 0` 作为唯一窗口阳性定义

当前 window binary target 是只要窗口内有任意 CpG island base 就为阳性。这会让边界窗口、只有很少 overlap 的窗口与完整 island 窗口同类，标签噪声高。

建议提供多种 window target：

1. **presence target**

   当前方案：`fraction > 0`。

   优点：召回敏感。
   缺点：边界噪声大。

2. **thresholded presence**

   例如 `fraction >= 0.05` 或 `fraction >= 0.10`。

   优点：降低微小 overlap 噪声。
   缺点：可能削弱边界召回。

3. **fraction regression**

   window head 直接预测 `fraction`。

   优点：与 base segmentation 自然一致。
   缺点：极度偏零，需要加权或 focal-like regression。

4. **ordinal bins**

   将 fraction 分桶：

   - 0
   - `(0, 0.05)`
   - `[0.05, 0.25)`
   - `[0.25, 0.75)`
   - `[0.75, 1]`

   优点：同时表达 presence 和强度。
   缺点：实现复杂度略高。

推荐方向：

- window head 输出两个量：
  - `window_presence_logit`
  - `window_fraction_logit`
- presence target 使用 `fraction >= threshold`。
- fraction target 使用 `smooth_l1` 或 BCE-style regression。

### 5.2 Window head 从 base prediction 派生

目前 window head 从独立 adapter 的 feature 做 attention pooling。为了减少任务对抗，可以把 window 目标更多绑定到 base prediction：

方案 A：纯派生

```text
window_score = mean(sigmoid(base_logits))
```

优点：最一致、无额外头。
缺点：窗口判别能力可能不足。

方案 B：校正派生

```text
window_score = MLP([mean(base_prob), max(base_prob), attention_pool(features)])
```

优点：保持一致性，同时保留窗口上下文。

方案 C：双路输出

```text
window_fraction = mean(base_prob)
window_presence = learned_window_head(features)
```

presence 用于分类指标，fraction 用于任务一致性。

推荐先做 B，因为它能减少冲突，同时不完全牺牲 window head 的表达力。

## 6. P3: 模型结构优化

### 6.1 多尺度 encoder

当前所有 ConvBlock 都在原始长度 `L` 上运行。CpG island 是长度通常数百到数千 bp 的区域，完全没必要所有层都 full resolution。

建议结构：

```text
seq -> stem conv
    -> stage1 full resolution
    -> downsample x2
    -> stage2 dilated conv
    -> downsample x2
    -> stage3 dilated conv
    -> upsample / skip
    -> base head
    -> window head on low-resolution feature
```

预期收益：

- 中高层计算量下降。
- window head 可直接使用低分辨率全局特征。
- base head 通过 skip 保持边界分辨率。

风险：

- 上采样可能损失边界精度。
- 需要验证 interval boundary error。

### 6.2 Depthwise separable Conv1d

将部分标准 Conv1d 替换为：

```text
depthwise Conv1d(groups=C)
pointwise Conv1d(1x1)
normalization
activation
```

适合大 channel 阶段，能显著降低 FLOPs。

建议实验：

- 只替换后两层高 channel block。
- 对比 full Conv 与 separable Conv 的 val PR-AUC、boundary error、step time。

### 6.3 BatchNorm 替换为 GroupNorm 或 LayerNorm1d

当前使用 `BatchNorm1d`。当 batch 很大时 BatchNorm 速度尚可，但在多 GPU、不同 batch size、混合采样比例下统计可能不稳定。

建议实验：

- `GroupNorm(num_groups=8 or 16)`。
- 或 ConvNeXt-style LayerNorm over channels。

如果未来做 DDP，多机多卡下 GroupNorm 更简单。

## 7. P4: 任务增强

### 7.1 Boundary auxiliary task

CpG island 预测最终要转成 interval，边界质量很重要。可以从 mask 派生 boundary label：

- island start/end 附近 `k` bp 为 boundary。
- 输出 `boundary_logits`。
- 损失使用 BCE/Focal。

好处：

- 让模型显式学习从非岛到岛的转换。
- 有助于 postprocess 的 interval 起止位置。

配置建议：

```yaml
training:
  lambda_boundary: 0.05
data:
  boundary_label_width: 32
```

### 7.2 Distance transform target

为每个 base 预测到最近 CpG island 边界或中心的距离，作为辅助回归任务。

优点：

- 比 hard boundary label 更平滑。
- 可以改善边界附近梯度。

缺点：

- 需要归一化距离。
- 对长窗口和跨 island 情况要定义清楚。

### 7.3 Hard negative curriculum

当前 hard negative 是固定比例。建议动态调整：

| 阶段 | positive | hard negative | random negative |
| --- | --- | --- | --- |
| early | 高 | 中 | 低 |
| middle | 中 | 高 | 中 |
| late | 接近真实分布 | 高 | 高 |

目标：

- 早期快速学会 island 内部特征。
- 中期强化 GC-rich non-island 区分。
- 后期校准真实分布下的 precision/recall。

## 8. 推荐实验矩阵

第一组：性能优先，低风险。

| 实验 | 改动 | 主要观察 |
| --- | --- | --- |
| E1 | GPU one-hot | step time、GPU 利用率 |
| E2 | bf16 + TF32 | 稳定性、速度 |
| E3 | uncertainty 替代 GradNorm | compile 是否恢复、收敛速度 |
| E4 | fast/full validation 拆分 | epoch wall time |

第二组：任务目标。

| 实验 | 改动 | 主要观察 |
| --- | --- | --- |
| E5 | window threshold 从 0 改 0.05/0.10 | window PR-AUC、base boundary recall |
| E6 | window fraction regression | fraction MAE、base PR-AUC |
| E7 | window score 派生自 base prob + learned correction | base/window 冲突是否下降 |
| E8 | consistency warmup | early training stability |

第三组：模型结构。

| 实验 | 改动 | 主要观察 |
| --- | --- | --- |
| E9 | depthwise separable high-channel blocks | FLOPs、PR-AUC |
| E10 | U-Net/多尺度 encoder | step time、boundary error |
| E11 | GroupNorm | batch size 敏感性 |
| E12 | boundary auxiliary task | interval F1、boundary error |

## 9. 判断优化是否成功

不要只看训练 loss。建议每个实验至少记录：

- windows/s 或 bases/s。
- GPU utilization。
- peak GPU memory。
- CPU utilization 和 dataloader wait time。
- val base PR-AUC。
- val base best F1。
- val window PR-AUC。
- val fraction MAE。
- interval-level F1。
- boundary mean absolute error。

性能优化合格标准：

- step time 至少下降 15%-20%，且 base PR-AUC 不下降超过 0.005-0.01。
- 或相同步时下可用更大 batch/window。

任务优化合格标准：

- interval F1 或 boundary error 改善。
- base/window 指标同时不明显退化。

## 10. 建议的实施顺序

1. 实现 GPU one-hot 或 embedding 输入，先解决 CPU batch 构造瓶颈。
2. 将 full config 默认多任务方法从 GradNorm 改为 uncertainty 或 fixed，并保留 GradNorm 作为实验配置。
3. 增加 fast/full validation 机制，降低每 epoch 验证成本。
4. 对 window target 做 threshold/fraction/ordinal 三组 ablation。
5. 实现 window score 从 base prob 派生的校正头。
6. 实现 depthwise separable block。
7. 若 boundary error 仍高，再加入 boundary auxiliary task。
8. 最后再考虑 U-Net/多尺度 encoder，因为它收益可能大，但改动和验证成本也最高。

