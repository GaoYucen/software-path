# DynaPath-LLM 小规模验证实验

本仓库用于验证 `pathllm.pdf` 中“道路文本/语义模态 + GPT-2 主干”是否能在旅行时间估计任务上跑通，并进一步观察加入动态交通状态后是否出现可解释收益。

当前版本的重点是一个可复现的小规模闭环：

```text
PKDD15 Porto GPS 轨迹
  -> OSM 近似 map matching
  -> OSM edge path tokens
  -> edge 文本/语义 embedding + 动态交通特征
  -> LSTM / Transformer / PathLLM-Static / DynaPath 变体对比
  -> LaTeX 论文表格与结果分析
```

## 当前保留的核心材料

```text
.
├── dynapath/                              # 数据集、模型、评估代码
├── scripts/
│   ├── prepare_pkdd15_osm_dynamic.py      # PKDD15 -> OSM edge 数据构建
│   ├── build_grid_semantic_embeddings.py  # 旧 grid token 文本 embedding 工具
│   ├── train_dynapath_llm.py              # DynaPath-LLM / GPT-2 训练入口
│   ├── train_neural_baselines.py          # LSTM / Transformer / PathLLM-Static
│   └── train_dynapath_variants.py         # 融合方式与动态模态消融
├── latex/
│   ├── latex.tex                          # 论文主文件
│   ├── latex.pdf                          # 当前编译结果
│   └── figure/                            # 论文图
├── reports/
│   ├── pathllm_small_scale_validation.md  # 小规模实验总结
│   ├── *_osm_5k_h128/*.json               # OSM 5K 主要结果
│   └── dynapath_llm_*/*.json              # GPT-2/LLM smoke 结果
├── pathllm.pdf                            # Path-LLM 参考论文
└── Path-LLM/                              # 原 Path-LLM 代码参考
```

大文件如 checkpoint、joblib 模型、OSM graph、processed npy 数据、缓存和虚拟环境不作为归档材料保留在 Git 中。

## OSM 数据构建

从 PKDD15 原始轨迹构建 OSM edge 级数据：

```bash
python scripts/prepare_pkdd15_osm_dynamic.py \
  --train-csv pkdd-15-predict-taxi-service-trajectory-i/train.csv \
  --out-dir data/processed/pkdd15_osm_5k \
  --graph-path data/raw/porto_drive.graphml \
  --max-rows 5000
```

主要输出：

```text
data/processed/pkdd15_osm_5k/
├── data_road.npy
├── dynamic_path.npy
├── trip_time.npy
├── row_num.npy
├── departure_time.npy
├── semantic_embeddings.npy
├── edge_metadata.csv
├── edge_texts.csv
├── trip_features.csv
└── dynamic_edge_features.csv
```

脚本逻辑包括：

- 下载或复用 Porto 区域 OSM drive graph。
- 将 GPS 点吸附到 OSM 节点。
- 用最短路桥接相邻 GPS 点，生成 edge 序列。
- 为道路段生成属性文本和语义 embedding。
- 使用出发时刻之前的历史窗口构造动态交通特征，避免未来信息泄漏。

## 模型训练

训练神经基线：

```bash
python scripts/train_neural_baselines.py \
  --data-dir data/processed/pkdd15_osm_5k \
  --output-dir reports/lstm_osm_5k_h128 \
  --models lstm \
  --hidden-size 128 \
  --epochs 20 \
  --batch-size 16
```

训练 PathLLM-Static：

```bash
python scripts/train_neural_baselines.py \
  --data-dir data/processed/pkdd15_osm_5k \
  --output-dir reports/pathllm_static_osm_5k_h128 \
  --models pathllm \
  --hidden-size 128 \
  --epochs 20 \
  --batch-size 16
```

训练 DynaPath 融合变体：

```bash
python scripts/train_dynapath_variants.py \
  --data-dir data/processed/pkdd15_osm_5k \
  --output-dir reports/simple_gate_osm_5k_h128 \
  --variant simple_gate \
  --hidden-size 128 \
  --epochs 20 \
  --batch-size 16
```

GPT-2 smoke test：

```bash
python scripts/train_dynapath_llm.py \
  --data-dir data/processed/pkdd15_osm_5k \
  --output-dir reports/dynapath_llm_osm_5k_smoke \
  --llm-name gpt2 \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 2 \
  --max-eval-batches 1
```

如只想验证模型形状和训练管线，可加 `--no-llm` 使用轻量 Transformer backbone。

## 当前结果

当前主要结果保存在：

```text
reports/pathllm_small_scale_validation.md
reports/lstm_osm_5k_h128/baseline_metrics.json
reports/transformer_osm_5k_h128/baseline_metrics.json
reports/pathllm_static_osm_5k_h128/baseline_metrics.json
reports/pathllm_static_osm_5k_no_text_h128/baseline_metrics.json
reports/static_only_osm_5k_h128/variant_metrics.json
reports/simple_gate_osm_5k_h128/variant_metrics.json
reports/concat_osm_5k_h128/variant_metrics.json
```

OSM 5K 小规模结果显示：道路文本/语义模态相对无文本版本有明显收益，但当前小数据和近似 map matching 下，LSTM 仍是最强基线。这个结论已经写入 `latex/latex.tex` 的实验分析部分。

## 论文编译

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

当前已生成：

```text
latex/latex.pdf
```

## 当前局限

- OSM map matching 是近似实现，尚不是工业级 HMM/Viterbi map matching。
- 语义 embedding 目前来自轻量文本模板/本地可复现编码，不等价于最终强文本编码器。
- 小规模 5K 实验主要用于验证趋势，不应作为最终论文大规模结论。
- GPT-2 smoke test 已验证主链路可跑，但完整 GPT-2 训练还受本地算力、缓存和样本规模限制。
