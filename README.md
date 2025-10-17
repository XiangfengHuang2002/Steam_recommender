# Steam Reccommender

## 1. 项目概述

本项目旨在构建一个混合型的 Steam 游戏推荐系统，以解决传统内容推荐的精度和泛化性问题。项目经历了从 **基础特征提取** 到 **内容相似度推荐**，再到 **基于 Triplet Loss 的深度学习 Embedding 建模** 的改进历程。

## 2. 数据准备与清洗 (`00_Data_Cleaning.ipynb`)

### 2.1 数据源与目标
* **数据源**: Kaggle Steam Games Dataset (`games.json`)
* **目标**: 过滤无效记录，处理缺失值，标准化关键字段，为后续特征工程打下基础。
* **结果**: 最终保留 **68,458 条** 有效数据（`steam_games_cleaned.csv`）。

### 2.2 关键清洗步骤
1.  **字段选择**: 保留了 `appID`, `name`, `price`, `tags`, `genres`, `positive`, `negative` 等核心字段。
2.  **缺失值与非空检查**: 重点保证标签、分类等文本字段的完整性。

## 3. 数据探索与衍生特征 (`01_Data_Exploration.ipynb` & `02_Feature_Engineering.ipynb`)

### 3.1 探索性分析
* **关联性**: 证实游戏的评论数、推荐数与好评率（`positive_ratio`）是衡量游戏流行度和质量的重要指标。

### 3.2 核心特征工程与改进

项目从基础特征衍生出以下关键特征，用于提升推荐的鲁棒性和准确性：

1.  **文本特征向量化 (Initial)**:
    * 将 `tags`, `categories`, `genres` 拼接为 `text_features`。
    * 使用 **TF-IDF** 对 `text_features` 进行向量化，作为 **内容推荐模块** 的输入。
2.  **加权分数 (Popularity Feature)**:
    * 结合好评数（`positive`）和评论总数（`total_reviews`），计算一个更稳定的 **加权分数** (`weighted_score`)，用于平衡游戏的质量和受欢迎程度，避免评论过少的高好评率游戏被过度推荐。
    * 公式示例：$$\text{WeightedScore} = \text{PositivePercent} \times \log(\text{TotalReviews} + 1)$$
3.  **高级分类特征 (Intent Category)**:
    * **（改进）** 虽然在数据清洗和基础特征工程 Notebook 中未直接展示生成过程，但在 **深度模型 (`04`)** 中，使用了 **`intent_category`** 字段进行 Embedding 可视化和模型训练。这表明项目在特征构建阶段引入了基于游戏核心玩法/意图的自动聚类（如 `Workflow.md` 中提到的 KMeans），以指导深度学习模型的相似性学习。

## 4. 推荐系统模块 I：基于内容的相似度推荐 (`03_Content_Recommendation.ipynb`)

### 4.1 模块定位
作为项目的 **基线 (Baseline)** 推荐方法，快速评估标签和分类特征的有效性。

### 4.2 推荐算法
* **特征**: TF-IDF 向量化的文本特征。
* **相似度**: **余弦相似度（Cosine Similarity）**。

### 4.3 成果
通过 `recommend_game` 函数，能够基于标签重叠度为游戏找到 Top K 相似游戏，例如为 `Stardew Valley` 推荐与其标签高度重合的游戏。

## 5. 推荐系统模块 II：深度相似度建模（改进核心） (`04_Siamese_Triplet_Model.ipynb`)

### 5.1 改进目标
克服传统 TF-IDF 推荐中 **标签稀疏** 和 **语义鸿沟** 问题，学习一个更具判别性的游戏 Embedding 空间，使**核心玩法相似**的游戏在向量空间中距离更近。

### 5.2 模型架构
* **网络**: 使用共享权重的 MLP（多层感知机）构建 **Siamese/Triplet Network**。
* **输入**: 经过标准化处理的融合特征（包括数值特征和文本 Embedding，可能包括 `tags`, `genres` 等）。
* **输出**: 游戏 Embedding 向量。

### 5.3 训练与损失函数
* **训练目标**: 相似游戏 Embedding 距离近，不相似游戏距离远。
* **核心损失**: **Triplet Loss**。
    $$L(A, P, N) = \max(\|\mathbf{f}(A) - \mathbf{f}(P)\|_2^2 - \|\mathbf{f}(A) - \mathbf{f}(N)\|_2^2 + \text{margin}, 0)$$
    其中 $A$ (Anchor)、$$P$$ (Positive，相似游戏)、$$N$$ (Negative，不相似游戏) 的构造，需结合游戏的**`intent_category`**和标签相似性来确定。

### 5.4 模型验证（Embedding 可视化）
* 通过 **t-SNE 降维** 将训练后的 Embedding 映射到 2D 空间。
* 可视化结果（散点图）使用 **`intent_category`** 进行着色，成功证明模型能够将具有相似**核心意图/玩法**的游戏聚集在一起，验证了 Triplet Loss 在精细化相似度学习上的有效性。

## 附录：技术栈

* **Python 库**: `pandas`, `numpy`, `sklearn` (TF-IDF, StandardScaler, TSNE)
* **深度学习框架**: `torch` 或 `tf_keras` (用于构建和训练 Triplet Network)
* **向量检索**: 训练后的 Embedding 可配合 **Faiss** 等工具进行高效的向量检索，构建最终推荐列表。
