# VulHME

Official implementation of **VulHME: A Heterogeneous Multi-scale Framework with Expert Learning for Code Vulnerability Detection**.

## Paper-Algorithm Mapping

- **Graph construction**: source code is expected to be converted to CPG-style heterogeneous graphs with node features and typed directed edges.
- **Local encoder**: Initial-Residual Relational GCN (IRR-GCN), implemented in `vulhme/layers.py`.
- **Global encoder**: Graphormer-style full-graph attention with spatial and edge-type structural bias, implemented in `vulhme/layers.py`.
- **Fusion**: `H_fused = H_local + beta * LayerNorm(H_global)`, where `H_global` is computed from `detach(H_local)`.
- **MACE**: three expert heads trained with CE, LDAM, and Focal losses, then aggregated by a supervised gating network.
- **Default paper setting**: 4 local layers, 1 global layer, 10 attention heads, `lambda=0.8`.

## Repository Layout

```text
VulHME/
  configs/                 Paper-style training configs
  scripts/                 Convenience shell commands
  vulhme/
    layers.py              IRR-GCN and Graphormer components
    losses.py              CE, LDAM, Focal, and MACE loss
    model.py               VulHME model
    train.py               Minimal training entry point
    metrics.py             Accuracy/precision/recall/F1/AUC helpers
```

## Quick Start

Install dependencies:

```bash
pip install torch dgl scikit-learn pandas pyyaml
```

Run with a JSONL graph dataset:

```bash
python -m vulhme.train --config configs/vulhme_reveal.yaml
```

Each JSONL line is one function graph:

```json
{
  "node_features": [[0.1, 0.2], [0.3, 0.4]],
  "edge_index": [[0, 1], [1, 0]],
  "edge_type": [0, 2],
  "label": 1,
  "spatial_pos": [[0, 1], [1, 0]],
  "edge_bias": [[0.0, 0.2], [0.1, 0.0]]
}
```
