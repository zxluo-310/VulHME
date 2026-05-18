from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRRGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.rel_weights = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)
        )
        self.self_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.initial_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index
        messages = x.new_zeros(num_nodes, self.hidden_dim)

        for relation in range(self.num_relations):
            mask = edge_type == relation
            if not torch.any(mask):
                continue
            rel_src = src[mask]
            rel_dst = dst[mask]
            transformed = self.rel_weights[relation](x[rel_src])
            degree = torch.bincount(rel_dst, minlength=num_nodes).clamp_min(1).to(x.dtype)
            relation_messages = x.new_zeros(num_nodes, self.hidden_dim)
            relation_messages.index_add_(0, rel_dst, transformed)
            messages = messages + relation_messages / degree.unsqueeze(-1)

        local_message = messages + self.self_weight(x)
        eta = self.gate(x0)
        out = (1.0 - eta) * local_message + eta * self.initial_weight(x0)
        out = self.norm(out)
        return F.relu(self.dropout(out))


class IRRGCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, num_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            IRRGCNLayer(hidden_dim, num_relations, dropout=dropout) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x0 = x
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, x0)
        return x


@dataclass
class GraphStructure:
    batch: torch.Tensor
    spatial_pos: Optional[torch.Tensor] = None
    edge_bias: Optional[torch.Tensor] = None


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 10, dropout: float = 0.1, max_spatial: int = 64) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_bias = nn.Embedding(max_spatial + 1, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _attention_one_graph(
        self,
        x: torch.Tensor,
        spatial_pos: Optional[torch.Tensor],
        edge_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        qkv = self.qkv(self.norm1(x)).view(num_nodes, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scores = torch.einsum("nhd,mhd->hnm", q, k) * self.scale

        if spatial_pos is not None:
            clipped = spatial_pos.clamp(min=0, max=self.spatial_bias.num_embeddings - 1)
            scores = scores + self.spatial_bias(clipped).squeeze(-1).unsqueeze(0)
        if edge_bias is not None:
            scores = scores + edge_bias.unsqueeze(0)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.einsum("hnm,mhd->nhd", weights, v).reshape(num_nodes, self.hidden_dim)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor, structure: GraphStructure) -> torch.Tensor:
        outputs = []
        for graph_id in torch.unique_consecutive(structure.batch):
            idx = torch.nonzero(structure.batch == graph_id, as_tuple=False).flatten()
            sub_x = x[idx]
            sub_spatial = None
            sub_edge_bias = None
            if structure.spatial_pos is not None:
                sub_spatial = structure.spatial_pos[idx][:, idx]
            if structure.edge_bias is not None:
                sub_edge_bias = structure.edge_bias[idx][:, idx]
            outputs.append(self._attention_one_graph(sub_x, sub_spatial, sub_edge_bias))

        attn_out = torch.cat(outputs, dim=0)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GraphormerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 10,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_spatial: int = 64,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            GraphormerLayer(hidden_dim, num_heads, dropout=dropout, max_spatial=max_spatial)
            for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, structure: GraphStructure) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, structure)
        return x


def mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
    pooled = x.new_zeros(batch_size, x.size(-1))
    pooled.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=batch_size).clamp_min(1).to(x.dtype)
    return pooled / counts.unsqueeze(-1)
