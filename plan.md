# 基于神经网络的 CpG Island 识别算法计划

## Tasks

- 报告中应明确说明具体操作时使用的输入与输出数据及其获取方法；
- 报告中需附上全部实现源码及其编译/运行方法，并进行实测检验。

## Summary

目标是实现一个可复现的 CpG island 识别流程：从参考基因组序列中切分窗口，训练神经网络对窗口内每个碱基进行 CpG island / non-CpG island 分割，同时用一个窗口级辅助输出头预测窗口内 CpG island 的存在强度，再把逐碱基预测结果合并为基因组区间。报告需明确输入、输出、数据获取方法、源代码、运行方法和实测验证。

数据源默认使用人类参考基因组 GRCh38/hg38：Ensembl 提供 GRCh38.p14 FASTA 下载入口，UCSC hg38 数据库中包含 `cpgIslandExt.txt.gz` 标注文件，可作为监督标签来源。参考链接：[Ensembl Human GRCh38.p14](https://www.ensembl.org/Homo_sapiens/)；[UCSC hg38 database index](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/database/)。

需要注意：UCSC CpG island 标注本身与传统规则定义高度相关，因此本项目的监督学习目标应表述为“学习并复现 UCSC hg38 CpG island 标注模式”，而不是证明神经网络发现了独立于规则定义的新生物学标准。模型若优于传统规则 baseline，需要进一步通过边界精度、困难负样本表现和跨染色体泛化来解释。

## Key Changes / Implementation Plan

- 训练数据：
  - 输入序列：GRCh38/hg38 染色体 FASTA，只保留 `A/C/G/T`，含 `N` 的窗口丢弃。
  - 逐碱基标签：UCSC `cpgIslandExt.txt.gz` 中的 CpG island 区间作为监督标签；窗口内每个碱基生成一个二值 mask，落在 CpG island 内标为 `1`，否则标为 `0`。
  - 窗口级辅助标签：由逐碱基 mask 派生，例如 `cpg_fraction = mask.mean()` 表示窗口内 CpG island 覆盖比例；也可同时派生 `has_cpg = 1[cpg_fraction > 0]` 表示窗口内是否存在 CpG island。
  - 采样策略：保留覆盖 CpG island 的正窗口，并从非 CpG island 区域采样负窗口；负窗口应包含随机负样本和高 GC、靠近 CpG island 边界的困难负样本，避免模型只学习简单 GC 差异。
  - 负样本比例：训练时可控制正窗口、随机负窗口、困难负窗口的采样比例；验证和测试阶段应尽量按完整染色体滑窗评估，避免只在人工平衡的数据集上报告过高指标。
  - 数据划分按染色体完成，避免相邻窗口泄漏：基础方案为训练集 `chr1-chr16`，验证集 `chr17-chr18`，测试集 `chr19-chr22`。
  - 数据划分斟酌：`chr19-chr22` 的 CpG island 密度和基因密度不一定代表全基因组，尤其 chr19 通常 CpG/GC 更富集。若时间允许，应增加一次替代染色体划分或 chromosome-level cross-validation，并报告均值和方差；若只使用固定划分，需要在报告中说明代表性限制。
  - 窗口设置：默认 512 bp 窗口，128 bp stride；不再按窗口重叠比例丢弃中间样本，而是保留所有不含 `N` 的窗口，用逐碱基 mask 表达部分重叠和边界信息。
  - 窗口参数消融：建议比较 `256/512/1024 bp` 窗口和 `64/128/256 bp` stride。较大窗口提供更完整的 GC/CpG 上下文但边界更粗，较小窗口有利于边界定位但可能降低整体判别稳定性。

- 模型结构：
  - 输入：`512 x 4` one-hot DNA 序列编码。
  - 主模型：多任务 1D CNN 分割模型，由共享序列编码器、逐碱基分割头和窗口级全局输出头组成。
  - 共享编码器：使用保持序列长度的 `Conv1D` 模块，例如 `Conv1D(64, k=7, padding=same)`、`Conv1D(128, k=5, padding=same)`、`Conv1D(256, k=3, padding=same)`，每层后接 BatchNorm、ReLU、Dropout；可使用 dilated convolution 扩大感受野。若使用 MaxPool，需要配套上采样或 U-Net 式 skip connection 恢复到 512 bp 分辨率。
  - 逐碱基分割头：在共享特征上接 `Conv1D(1, k=1)` 和 Sigmoid，输出长度为 `512` 的概率向量，表示每个碱基属于 CpG island 的概率 `p_base ∈ [0,1]`。
  - 窗口级全局输出头：对共享特征做 Global Average Pooling 或 attention pooling，再接 Dropout、全连接层、Sigmoid/linear 输出；推荐预测 `cpg_fraction ∈ [0,1]`，也可附加预测 `has_cpg ∈ {0,1}` 作为窗口内是否存在 CpG island 的辅助任务。
  - 最终输出：模型同时输出逐碱基概率 `p_base[512]` 和窗口级存在强度 `p_window`；区间生成主要依赖逐碱基概率，窗口级输出用于辅助训练、过滤低置信窗口或报告窗口级指标。
  - 对照基线：传统规则法，使用长度、GC 含量、Obs/Exp CpG 比值阈值作为 baseline；另外可实现基于窗口特征的 Logistic Regression / Random Forest / XGBoost，以及 HMM 方法作为对照。XGBoost 和 HMM 属于可选扩展，若时间有限，至少保证传统规则 baseline 和一个简单机器学习 baseline。

- 训练任务：
  - 损失函数：多任务损失 `L = L_base + λ * L_window`。`L_base` 为逐碱基 Binary Cross Entropy，可加入 Dice loss / focal loss 或正负碱基权重缓解类别不平衡；`L_window` 根据辅助标签选择，预测 `cpg_fraction` 时用 MSE/BCE，预测 `has_cpg` 时用 Binary Cross Entropy。
  - 优化器：AdamW，初始学习率 `1e-3`，batch size `128` 或 `256`。
  - 停止策略：以验证集碱基层 PR-AUC、碱基层 F1 或区间级 F1 为主指标，连续若干轮不提升即 early stopping；窗口级辅助指标作为诊断项。
  - 阈值选择：在验证集上分别选择逐碱基概率阈值和可选的窗口级过滤阈值，而不是固定 0.5。
  - 区间生成：测试/预测时滑窗扫描整条染色体，对重叠窗口的逐碱基概率按平均值或加权平均聚合为全染色体碱基层概率轨道；对概率轨道阈值化后合并相邻阳性碱基，允许设置最大 gap，过滤长度 `<200 bp` 的短区间，输出 BED 格式。
  - 后处理参数：逐碱基阈值、窗口级过滤阈值、最大 gap、最小长度、score 聚合方式都应只在验证集上选择；测试集只用于最终一次评估，避免后处理过拟合测试集。

- 输入 / 输出接口：
  - 训练输入：`genome.fa`、`cpgIslandExt.txt.gz`、配置文件。
  - 训练输出：模型权重、训练日志、逐碱基与窗口级验证指标、逐碱基阈值、可选窗口级过滤阈值。
  - 预测输入：待识别 FASTA。
  - 预测输出：`predicted_cpg_islands.bed`，字段包含 `chrom start end score`；可选输出逐碱基概率轨道 `predicted_cpg_signal.bedGraph`。
  - 评估输出：碱基层 ROC-AUC、PR-AUC、Accuracy、Precision、Recall、F1；窗口级 `cpg_fraction` 或 `has_cpg` 指标；区间级预测与 UCSC 标注的 overlap、IoU、precision、recall、F1 和边界误差统计。
  - 指标解读：Accuracy 只作为辅助指标，主指标优先使用 PR-AUC、F1、precision、recall 和区间级 IoU/overlap。由于非 CpG island 碱基占多数，单纯 accuracy 可能高估模型效果。

## Test Plan

- 数据处理测试：
  - 检查 FASTA 读取、窗口切分、one-hot 编码维度是否正确。
  - 检查逐碱基 mask 是否与 BED 区间坐标一致，尤其注意 BED 的 0-based、half-open 坐标约定。
  - 检查窗口级 `cpg_fraction` / `has_cpg` 是否由逐碱基 mask 正确派生。
  - 检查训练/验证/测试染色体无交叉。

- 模型训练测试：
  - 用小规模染色体片段跑通完整训练流程。
  - 验证总 loss、逐碱基 loss 和窗口级辅助 loss 下降，模型输出概率范围为 `[0,1]`。
  - 检查逐碱基输出维度为 `batch x 512`，窗口级输出维度为 `batch x 1`。
  - 保存并重新加载模型后，预测结果一致。

- 实测验证：
  - 在测试染色体上与 UCSC CpG island 标注比较。
  - 报告 CNN 与传统规则 baseline 的指标对比。
  - 报告随机负样本和困难负样本上的分层表现，尤其关注高 GC 非 CpG island 区域和 CpG island 边界附近的误报/漏报。
  - 报告窗口长度、stride、损失组合和后处理参数的消融实验；若算力有限，至少比较默认模型与一个较小窗口或无窗口级辅助头的版本。
  - 展示 1-2 个染色体局部区域的真实标注和预测区间对照图。

## Report Organization

- 引言：CpG island 生物学意义、传统识别规则、神经网络方法动机。
- 数据与预处理：说明 FASTA 和 UCSC 标注文件来源、下载方法、窗口生成、标签定义、数据划分。
- 模型方法：说明 one-hot 编码、CNN 结构、损失函数、训练超参数、预测区间合并策略。
- 实验设计：说明 baseline、碱基层/窗口级/区间级评价指标、验证集阈值选择、测试集设置。
- 结果与分析：展示指标表、训练曲线、baseline 对比、消融实验、困难负样本表现、错误案例分析。
- 运行说明：列出环境依赖、训练命令、预测命令、输出文件格式。
- 附录：附全部源代码，或说明代码文件结构与主要函数。

## Assumptions

- 默认识别对象为人类基因组 CpG island，而不是其他物种。
- 默认使用 UCSC hg38 CpG island 标注作为监督学习标签。
- 默认实现重点是课程/报告级可复现实验，不追求生产级全基因组在线服务。
- 若算力有限，可只用部分染色体训练，但测试集仍需独立染色体并在报告中说明。
