from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class GraphSample:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    label: int
    spatial_pos: torch.Tensor | None = None
    edge_bias: torch.Tensor | None = None


class JsonlGraphDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.samples = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        item = self.samples[idx]
        edge_index = torch.tensor(item["edge_index"], dtype=torch.long)
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        return GraphSample(
            node_features=torch.tensor(item["node_features"], dtype=torch.float),
            edge_index=edge_index,
            edge_type=torch.tensor(item["edge_type"], dtype=torch.long),
            label=int(item["label"]),
            spatial_pos=torch.tensor(item["spatial_pos"], dtype=torch.long) if "spatial_pos" in item else None,
            edge_bias=torch.tensor(item["edge_bias"], dtype=torch.float) if "edge_bias" in item else None,
        )


def collate_graphs(samples: list[GraphSample]) -> dict[str, torch.Tensor | None]:
    node_features = []
    edge_indices = []
    edge_types = []
    batch = []
    labels = []
    spatial_blocks = []
    edge_bias_blocks = []
    offset = 0
    has_spatial = all(sample.spatial_pos is not None for sample in samples)
    has_edge_bias = all(sample.edge_bias is not None for sample in samples)

    for graph_id, sample in enumerate(samples):
        num_nodes = sample.node_features.size(0)
        node_features.append(sample.node_features)
        edge_indices.append(sample.edge_index + offset)
        edge_types.append(sample.edge_type)
        batch.append(torch.full((num_nodes,), graph_id, dtype=torch.long))
        labels.append(sample.label)
        if has_spatial:
            spatial_blocks.append(sample.spatial_pos)
        if has_edge_bias:
            edge_bias_blocks.append(sample.edge_bias)
        offset += num_nodes

    result = {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_type": torch.cat(edge_types, dim=0),
        "batch": torch.cat(batch, dim=0),
        "targets": torch.tensor(labels, dtype=torch.long),
        "spatial_pos": None,
        "edge_bias": None,
    }

    if has_spatial:
        result["spatial_pos"] = torch.block_diag(*spatial_blocks)
    if has_edge_bias:
        result["edge_bias"] = torch.block_diag(*edge_bias_blocks)
    return result


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
