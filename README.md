# DynaPath-Lite 实验说明

本仓库用于支撑论文《面向动态交通场景的静态-动态解耦多模态路径表示学习方法》的第一版实验。当前目标不是完整复刻道路段级大模型路径表示，而是先在真实出租车轨迹数据上跑通一个可复现、无未来信息泄漏的动态路径表示验证闭环。

## 实验核心思路

论文关注的问题是旅行时间估计：给定路径 `P` 和出发时刻 `tau`，预测该次出行的旅行时间。

当前实验将路径信息拆为两类：

- 静态信息：路径长度、唯一网格数量、累计距离、起终点坐标、时间上下文、呼叫类型等。
- 动态信息：出发时刻之前历史窗口内的速度、速度波动、观测数量、速度比、时效性和可靠性等。

由于 PKDD15 Porto 数据提供 GPS 折线但不提供道路段 ID，当前版本使用网格单元作为快速路径 token：

```text
GPS polyline -> grid-cell path tokens -> static features + dynamic features -> TTE model
```

这个设计对应论文中的 DynaPath-Lite：先验证静态-动态解耦组织和动态交通状态是否有价值，再在后续版本中替换为 OSM 地图匹配道路段、道路文本语义和序列神经编码器。

## 目录结构

```text
.
├── scripts/
│   ├── prepare_pkdd15_grid_dynamic.py      # 构建 PKDD15 网格路径和动态特征
│   ├── train_pkdd15_dynapath_lite.py       # 训练静态/静态+动态 TTE 基线
│   ├── run_pkdd15_ablation.py              # 动态特征消融与稀疏退化实验
│   ├── train_dynapath_llm.py               # Path-LLM 风格大模型训练入口
│   ├── train_neural_baselines.py           # 神经基线训练（LSTM/Transformer/PathLLMStatic）
│   ├── train_dynapath_variants.py          # DynaPathLLM 变体训练与架构消融
│   ├── run_full_experiments.py             # 统一实验管理与 LaTeX 表格生成
│   ├── analyze_results.py                  # 结果分析与论文图表生成
│   ├── prepare_geolife_smoke.py            # GeoLife 小样例预处理
│   └── smoke_test_dynamic.py               # 动态构建 smoke test
├── dynapath/
│   ├── models.py                           # DynaPathLLM 完整模型（TPfusion + ReliabilityAwareFusion）
│   ├── baselines.py                        # 神经基线模型（LSTM, Transformer, PathLLMStatic）
│   ├── data.py                             # NPY 路径数据集封装
│   └── eval.py                             # 统一评估工具（指标、bootstrap CI）
├── data/
│   ├── raw/                                # 小样例原始数据
│   └── processed/                          # 处理后的路径 token、特征和标签
├── reports/                                # 实验指标、图表、LaTeX 表格和模型文件
│   ├── figures/                            # 论文分析图表（5张）
│   └── tables/                             # 自动生成的 LaTeX 对比表
├── latex/                                  # 软件学报论文 LaTeX 源文件
├── pkdd-15-predict-taxi-service-trajectory-i/
│   └── train.csv                           # PKDD15 原始训练数据
└── Path-LLM/                               # 原 Path-LLM 代码参考
```

## 数据构建

输入文件：

```text
pkdd-15-predict-taxi-service-trajectory-i/train.csv
```

核心处理步骤：

1. 读取前 120000 条 PKDD15 训练记录。
2. 删除缺失轨迹、空轨迹、过短轨迹和明显越界轨迹。
3. 将 GPS 点量化为 `0.001` 度网格单元。
4. 合并连续重复网格，得到路径 token 序列。
5. 按出发时间排序，使用历史窗口 `[tau - 60 min, tau)` 增量构建动态特征。
6. 当前 trip 的观测只在其特征计算完成后加入历史队列，避免未来信息泄漏。

复现命令：

```bash
python scripts/prepare_pkdd15_grid_dynamic.py \
  --train-csv pkdd-15-predict-taxi-service-trajectory-i/train.csv \
  --max-rows 120000 \
  --out-dir data/processed/pkdd15_grid_120k_clean
```

主要输出：

```text
data/processed/pkdd15_grid_120k_clean/
├── trip_features.csv          # 静态 + 动态表格特征
├── data_road.npy              # 网格路径 token 序列
├── dynamic_path.npy           # token 级动态特征张量
├── trip_time.npy              # 旅行时间标签
├── row_num.npy                # 有效 token 长度
├── departure_time.npy         # 出发时间
├── dynamic_cell_features.csv  # token 级动态明细
└── metadata.json              # 数据构建元信息
```

## OSM 道路段数据

为了从网格 token 升级到真实道路段，本仓库新增了 OSM 近似 map matching 脚本：

```bash
python scripts/prepare_pkdd15_osm_dynamic.py \
  --train-csv pkdd-15-predict-taxi-service-trajectory-i/train.csv \
  --out-dir data/processed/pkdd15_osm_5k \
  --graph-path data/raw/porto_drive.graphml \
  --max-rows 5000
```

该脚本会：

- 下载或复用 Porto 区域 OSM drive graph；
- 将 GPS 点吸附到 OSM 路网节点，并用最短路桥接相邻点；
- 导出真实 OSM edge 序列；
- 生成 `edge_metadata.csv`、`edge_texts.csv` 和 `semantic_embeddings.npy`；
- 生成与现有训练脚本兼容的 `data_road.npy`、`dynamic_path.npy`、`trip_time.npy` 等文件。

当前已验证可用的目录示例：

```text
data/processed/pkdd15_osm_smoke_500/
data/processed/pkdd15_osm_5k/
```

当前数据摘要：

| 项目 | 数值 |
|---|---:|
| 读取原始记录数 | 119999 |
| 越界剔除轨迹数 | 734 |
| 可用路径样本数 | 113660 |
| 网格路径单元数 | 20227 |
| 最大路径 token 长度 | 64 |
| 动态历史窗口 | 60 分钟 |
| 训练/验证/测试划分 | 79562/17049/17049 |

## 主实验

训练命令：

```bash
python scripts/train_pkdd15_dynapath_lite.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --report reports/pkdd15_dynapath_lite_120k_noleak_metrics.json \
  --model-dir reports/pkdd15_dynapath_lite_120k_noleak_models
```

注意：PKDD15 的完整 trip 旅行时间等于 `(len(POLYLINE) - 1) * 15` 秒，因此 `num_points` 会直接泄漏标签。训练脚本已显式排除该字段。

主实验结果：

| 模型 | MAE/s | RMSE/s | MAPE | MARE |
|---|---:|---:|---:|---:|
| Train mean | 268.14 | 353.38 | 0.5396 | 0.4036 |
| Ridge static | 97.42 | 152.09 | 0.1580 | 0.1466 |
| Ridge static+dynamic | 91.79 | 146.81 | 0.1473 | 0.1382 |
| HGB static | 88.60 | 140.46 | 0.1417 | 0.1333 |
| HGB DynaPath-Lite | **85.22** | **137.38** | **0.1357** | **0.1283** |
| RF DynaPath-Lite | 87.94 | 140.08 | 0.1404 | 0.1323 |

结论：动态特征在 Ridge 和 HGB 上均带来稳定收益，说明历史交通状态能够补充静态路径几何和时间上下文。

## 消融实验

消融命令：

```bash
python scripts/run_pkdd15_ablation.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --report reports/pkdd15_dynapath_lite_120k_ablation_metrics.json \
  --markdown reports/pkdd15_dynapath_lite_120k_ablation_report.md
```

消融结果：

| 实验 | 特征数 | MAE/s | RMSE/s | MARE |
|---|---:|---:|---:|---:|
| HGB static | 20 | 88.59 | 140.34 | 0.1333 |
| HGB full dynamic | 32 | **85.22** | **137.38** | **0.1283** |
| w/o speed | 28 | 86.72 | 138.88 | 0.1305 |
| w/o density | 30 | 85.40 | 137.36 | 0.1285 |
| w/o quality | 26 | 85.55 | 137.57 | 0.1288 |
| sparse-30% with quality | 32 | 88.41 | 140.72 | 0.1331 |
| sparse-30% w/o quality | 26 | 88.47 | 140.60 | 0.1331 |

说明：

- 速度中位数和速度比是当前动态模态中的主要增益来源。
- 观测数量、时效性和可靠性组成的动态质量信号带来小幅但稳定的辅助作用。
- 单独删除当前定义的 reliability 标量没有造成性能下降，说明在表格模型中观测数量和时效性已经提供了较强质量线索。
- 因此，当前实验支持“动态质量感知”的必要性；完整的可靠性门控效果还需要在道路段级序列模型中进一步验证。

## Path-LLM 风格大模型代码

根据 `pathllm.pdf` 和 `Path-LLM/models/Models.py`，Path-LLM 的关键结构是：

1. 使用拓扑 embedding 和文本 embedding 表示道路段。
2. 使用 TPalign 对齐拓扑和文本模态。
3. 使用 TPfusion 门控融合拓扑和文本表示。
4. 跳过 GPT-2 原始 token embedding，将融合后的路径 embedding 作为 `inputs_embeds` 输入 GPT-2。
5. 冻结 GPT-2 大部分参数，仅微调位置编码、LayerNorm 和任务头。

本仓库新增了对应的 DynaPath-LLM 风格模型代码：

```text
dynapath/models.py
```

相较 Path-LLM，当前模型额外接入动态交通模态：

```text
topology embedding + semantic/text embedding
        -> TPfusion static representation

dynamic traffic features
        -> DynamicStateEncoder

static representation + dynamic representation + reliability
        -> ReliabilityAwareFusion
        -> GPT-2 inputs_embeds
        -> TTE head
```

模型中包含三类损失：

- `loss_tte`：旅行时间估计 MSE 损失。
- `loss_ts_align`：道路级 Topology-Semantics 对齐，包含实例级和特征级 InfoNCE。
- `loss_sd_align`：路径级 Static-Dynamic 对齐，使用 batch 内静态路径表示和动态路径表示做 InfoNCE。

工程可跑版新增了一个轻量文本模态构建步骤：当只有网格 token 而没有 OSM 道路文本时，先根据 token 的历史速度、可靠性、出现频率和时段分布自动生成文本描述，再编码为 `semantic_embeddings.npy`。

先生成文本模态：

```bash
python scripts/build_grid_semantic_embeddings.py \
  --data-dir data/processed/pkdd15_grid_120k_clean
```

训练入口：

```bash
python scripts/train_dynapath_llm.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/dynapath_llm_debug \
  --llm-name gpt2 \
  --epochs 3 \
  --batch-size 8
```

脚本会自动读取 `data-dir/semantic_embeddings.npy`。如果只想做 GPT-2 工程 smoke test，可使用较小数据集和 batch 限制：

```bash
python scripts/train_dynapath_llm.py \
  --data-dir data/processed/pkdd15_quick_5k \
  --output-dir reports/dynapath_llm_quick_smoke \
  --llm-name gpt2 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --max-train-batches 2 \
  --max-eval-batches 1
```

如果本机没有 GPT-2 缓存或 Transformers 环境，可以先用小 Transformer 做形状调试：

```bash
python scripts/train_dynapath_llm.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/dynapath_llm_debug_no_llm \
  --no-llm \
  --epochs 1 \
  --batch-size 8
```

注意：当前网格 token 版本的 `semantic_embeddings.npy` 来自自动生成的文本描述，适合工程验证“文本模态 + GPT-2 主链”是否能跑通，但它不是最终论文版的真实道路文本语义。最终版本仍需要 OSM 地图匹配、道路属性文本和更强的语义编码方式。

大模型训练依赖可参考：

```bash
pip install -r requirements-llm.txt
```

## 论文文件

论文主文件：

```text
latex/latex.tex
```

编译命令：

```bash
cd latex
xelatex -interaction=nonstopmode latex.tex
biber latex
xelatex -interaction=nonstopmode latex.tex
xelatex -interaction=nonstopmode latex.tex
```

输出：

```text
latex/latex.pdf
```

## 神经序列基线

新增了完整的神经网络基线模型（`dynapath/baselines.py`）和训练脚本：

```bash
# 训练 LSTM、Transformer、PathLLM-Static 三个基线
# --no-llm 模式使用小 Transformer 代替 GPT-2，无需安装 transformers
python scripts/train_neural_baselines.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/neural_baselines \
  --models lstm,transformer,pathllm \
  --no-llm --epochs 20 --batch-size 16
```

模型说明：

| 模型 | 结构 |
|---|---|
| LSTM | 可训练 token embedding + 2层 BiLSTM + 池化 + MLP |
| Transformer | token embedding + 位置编码 + 2层 TransformerEncoder + 池化 + MLP |
| PathLLM-Static | 拓扑/semantic embedding + TPFusion 门控 + backbone (GPT-2/Transformer) |

## DynaPathLLM 变体与架构消融

支持 6 种模型变体的训练（`scripts/train_dynapath_variants.py`），用于验证论文中的架构设计：

```bash
# 训练完整模型和所有消融变体
python scripts/train_dynapath_variants.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/dynapath_variants \
  --variant full,no_align,no_sd_align,concat,simple_gate,static_only \
  --no-llm --epochs 20 --batch-size 16
```

| 变体 | 说明 |
|---|---|
| `full` | 完整 DynaPathLLM：TPfusion + DynamicEncoder + ReliabilityAwareFusion + TS/SD 对齐 |
| `no_align` | 完整架构但关闭所有对齐损失（λ_TS=λ_SD=0） |
| `no_sd_align` | 关闭静态-动态对齐（λ_SD=0） |
| `concat` | 简单拼接融合替代可靠性感知门控 |
| `simple_gate` | 普通门控融合（无显式可靠性输入） |
| `static_only` | 纯静态：TPfusion + backbone，无动态模态 |

## 统一实验管理

一键运行所有实验层级：

```bash
# 快速验证管线（小 epoch，仅核心变体）
python scripts/run_full_experiments.py --mode quick

# 完整论文实验（所有模型和变体）
python scripts/run_full_experiments.py --mode full

# 仅从已有结果生成对比表
python scripts/run_full_experiments.py --compare-only
```

生成产物：
- `reports/full_experiment_comparison.json` — 统一对比数据
- `reports/tables/full_comparison.tex` — 全模型对比 LaTeX 表
- `reports/tables/fusion_comparison.tex` — 融合策略对比 LaTeX 表
- `reports/tables/arch_ablation.tex` — 架构消融 LaTeX 表

## 结果分析与可视化

生成论文用的分析图表：

```bash
python scripts/analyze_results.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/figures/
```

生成 5 张论文图表：
1. `error_cdf.png` — 静态 vs 动态模型误差累积分布对比
2. `reliability_distribution.png` — 可靠性得分分布（直方图 + CDF）
3. `performance_by_length.png` — 不同路径长度下的 MAE 对比
4. `performance_by_hour.png` — 不同时段的模型表现
5. `dynamic_gain_scatter.png` — 动态增益 vs 标签/可靠性散点图

## 当前局限

- 网格 token 只是快速替代方案，不等价于真实道路段。
- 当前环境尚未安装 PyTorch 和 HuggingFace Transformers，序列模型的端到端训练需要在配置好环境的机器上运行。
- 后续需要引入 OSM 地图匹配、道路属性文本、道路段拓扑和序列神经编码器，形成完整的拓扑-语义-动态三模态模型。
- 当前使用可训练的随机初始化 embedding 作为 topology 和 semantic 模态的占位表示，实际场景中应替换为 node2vec 拓扑嵌入和道路文本嵌入。

## 推荐复现顺序

```bash
# 1. 构建数据
python scripts/prepare_pkdd15_grid_dynamic.py \
  --train-csv pkdd-15-predict-taxi-service-trajectory-i/train.csv \
  --max-rows 120000 \
  --out-dir data/processed/pkdd15_grid_120k_clean

# 2. 跑表格基线实验
python scripts/train_pkdd15_dynapath_lite.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --report reports/pkdd15_dynapath_lite_120k_noleak_metrics.json \
  --model-dir reports/pkdd15_dynapath_lite_120k_noleak_models

# 3. 跑消融实验
python scripts/run_pkdd15_ablation.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --report reports/pkdd15_dynapath_lite_120k_ablation_metrics.json \
  --markdown reports/pkdd15_dynapath_lite_120k_ablation_report.md

# 4. [需要 torch] 跑神经基线
python scripts/train_neural_baselines.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/neural_baselines \
  --models lstm,transformer,pathllm \
  --no-llm --epochs 20 --batch-size 16

# 5. [需要 torch] 跑 DynaPathLLM 变体消融
python scripts/train_dynapath_variants.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/dynapath_variants \
  --variant full,no_align,concat,simple_gate,static_only \
  --no-llm --epochs 20 --batch-size 16

# 6. 生成分析图表
python scripts/analyze_results.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --output-dir reports/figures/

# 7. 生成对比表
python scripts/run_full_experiments.py --compare-only

# 8. 编译论文
cd latex
xelatex -interaction=nonstopmode latex.tex
biber latex
xelatex -interaction=nonstopmode latex.tex
xelatex -interaction=nonstopmode latex.tex
```
