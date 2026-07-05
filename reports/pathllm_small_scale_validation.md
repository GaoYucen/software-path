# PathLLM Small-Scale Validation

## Goal

Use small-scale data to check whether PathLLM-style effects can already be observed before moving to larger experiments.

## Existing 5K Grid Results

Source:

- `reports/neural_baselines_quick/baseline_metrics.json`
- `reports/dynapath_variants_quick/variant_metrics.json`

Key results:

| Setting | MAE (s) | Interpretation |
|---|---:|---|
| LSTM | 346.59 | Best small-data sequence baseline on grid tokens |
| DynaPathLLM-SimpleGate | 500.76 | Dynamic modality helps relative to static-only |
| DynaPathLLM-StaticOnly | 513.06 | Worse than SimpleGate |
| DynaPathLLM-Concat | 515.27 | Worse than SimpleGate |
| DynaPathLLM-Full | 524.18 | Worst among variants on small data |

Observed effects on grid 5K:

- Dynamic modality helps: `SimpleGate 500.76 < StaticOnly 513.06`
- Gated fusion beats concat: `SimpleGate 500.76 < Concat 515.27`
- Full alignment + reliability gating does not help on small data: `Full 524.18 > SimpleGate 500.76`

## New 5K OSM Results

Source:

- `reports/lstm_osm_5k_h128/baseline_metrics.json`
- `reports/transformer_osm_5k_h128/baseline_metrics.json`
- `reports/pathllm_static_osm_5k_h128/baseline_metrics.json`
- `reports/pathllm_static_osm_5k_no_text_h128/baseline_metrics.json`
- `reports/static_only_osm_5k_h128/variant_metrics.json`
- `reports/concat_osm_5k_h128/variant_metrics.json`
- `reports/simple_gate_osm_5k_h128/variant_metrics.json`

Key results:

| Setting | MAE (s) | RMSE (s) | MAPE | MARE | Interpretation |
|---|---:|---:|---:|---:|---|
| LSTM | 190.85 | 247.07 | 0.3790 | 0.3263 | Best small-data sequence baseline |
| PathLLM-Static with OSM text | 281.41 | 394.00 | 0.4014 | 0.4812 | Real OSM text helps a lot |
| Transformer | 316.85 | 427.67 | 0.4500 | 0.5418 | Weaker than PathLLM-style text fusion |
| PathLLM-Static without text | 351.72 | 458.37 | 0.5083 | 0.6014 | Clear drop after removing text modality |
| DynaPathLLM-SimpleGate with OSM text | 294.48 | 406.84 | 0.4177 | 0.5035 | Dynamic gate is better than concat but worse than static-only |
| DynaPathLLM-Concat with OSM text | 300.65 | 412.80 | 0.4259 | 0.5141 | Simple concatenation is weaker than gating |

Observed effects on OSM 5K:

- Text modality helps strongly: `281.41 < 351.72`
- PathLLM-style text fusion beats the plain Transformer baseline: `281.41 < 316.85`
- PathLLM-style topology/text fusion is much healthier on OSM edges than on grid pseudo-tokens
- Dynamic modality is not yet helping in this current OSM 5K setup: `SimpleGate 294.48 > PathLLM-Static 281.41`
- Gated dynamic fusion is still better than concat: `SimpleGate 294.48 < Concat 300.65`

## Conclusion

Small-scale validation gives a mixed but useful answer:

- Yes, some PathLLM-style effects are visible already.
- The clearest confirmed effect is that real OSM text modality improves the PathLLM-style static model on 5K data.
- PathLLM-style text fusion also beats a plain Transformer baseline on the OSM edge sequence.
- The dynamic-fusion side is less stable: on old grid 5K, dynamic gate helps; on new OSM 5K, SimpleGate beats concat but remains worse than the static PathLLM-style model.
- Full multimodal alignment and reliability-aware fusion still do not look like small-data winners.
