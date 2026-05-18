from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .layers import GraphStructure, GraphormerEncoder, IRRGCNEncoder, mean_pool


@dataclass
class VulHMEConfig:
    input_dim: int = 100
    hidden_dim: int = 200
    num_classes: int = 2
    num_relations: int = 8
    local_layers: int = 4
    global_layers: int = 1
    global_heads: int = 10
    dropout: float = 0.1
    global_scale_init: float = 0.1


class ExpertHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VulHME(nn.Module):
    def __init__(self, config: VulHMEConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = (
            nn.Linear(config.input_dim, config.hidden_dim)
            if config.input_dim != config.hidden_dim
            else nn.Identity()
        )
        self.local_encoder = IRRGCNEncoder(
            hidden_dim=config.hidden_dim,
            num_relations=config.num_relations,
            num_layers=config.local_layers,
            dropout=config.dropout,
        )
        self.global_encoder = GraphormerEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.global_heads,
            num_layers=config.global_layers,
            dropout=config.dropout,
        )
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)
        self.global_gate = nn.Parameter(torch.zeros(1))
        self.global_scale = config.global_scale_init

        self.experts = nn.ModuleList(
            ExpertHead(config.hidden_dim, config.num_classes, config.dropout) for _ in range(3)
        )
        self.gate = nn.Sequential(
            nn.Linear(config.num_classes * 3, config.num_classes * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.num_classes * 2, 3),
        )

    def encode(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        spatial_pos: Optional[torch.Tensor] = None,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(node_features)
        local = self.local_encoder(x, edge_index, edge_type)
        structure = GraphStructure(batch=batch, spatial_pos=spatial_pos, edge_bias=edge_bias)
        global_features = self.global_encoder(local.detach(), structure)
        beta = torch.sigmoid(self.global_gate) * self.global_scale
        fused = local + beta * self.fusion_norm(global_features)
        return mean_pool(fused, batch)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        spatial_pos: Optional[torch.Tensor] = None,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        graph_repr = self.encode(node_features, edge_index, edge_type, batch, spatial_pos, edge_bias)
        expert_logits = [expert(graph_repr) for expert in self.experts]
        concat_logits = torch.cat(expert_logits, dim=-1)
        gate_weights = torch.softmax(self.gate(concat_logits), dim=-1)
        final_logits = torch.zeros_like(expert_logits[0])
        for idx, logits in enumerate(expert_logits):
            final_logits = final_logits + gate_weights[:, idx : idx + 1] * logits
        return {
            "logits": final_logits,
            "expert_logits": expert_logits,
            "gate_weights": gate_weights,
            "graph_repr": graph_repr,
        }
