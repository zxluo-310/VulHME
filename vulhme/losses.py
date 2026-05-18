from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    def __init__(
        self,
        cls_num_list: Sequence[int],
        max_m: float = 0.5,
        scale: float = 30.0,
        class_balance_beta: float = 0.9999,
    ) -> None:
        super().__init__()
        counts = torch.tensor(cls_num_list, dtype=torch.float)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts.clamp_min(1.0)))
        margins = margins * (max_m / margins.max())
        effective_num = 1.0 - torch.pow(torch.tensor(class_balance_beta), counts)
        weights = (1.0 - class_balance_beta) / effective_num.clamp_min(1e-12)
        weights = weights / weights.sum() * len(cls_num_list)
        self.register_buffer("margins", margins)
        self.register_buffer("weights", weights)
        self.scale = scale

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(logits.device)[targets]
        adjusted = logits.clone()
        adjusted[torch.arange(logits.size(0), device=logits.device), targets] -= margins
        return F.cross_entropy(self.scale * adjusted, targets, weight=self.weights.to(logits.device))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        target_log_prob = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_prob) ** self.gamma) * target_log_prob
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[targets]
        return loss.mean()


@dataclass
class MACELossOutput:
    total: torch.Tensor
    specialization: torch.Tensor
    aggregation: torch.Tensor
    ce: torch.Tensor
    ldam: torch.Tensor
    focal: torch.Tensor


class MACELoss(nn.Module):
    def __init__(
        self,
        cls_num_list: Sequence[int],
        lambda_agg: float = 0.8,
        focal_gamma: float = 2.0,
        ldam_max_m: float = 0.5,
        ldam_scale: float = 30.0,
        class_balance_beta: float = 0.9999,
    ) -> None:
        super().__init__()
        self.lambda_agg = lambda_agg
        counts = torch.tensor(cls_num_list, dtype=torch.float)
        inv = 1.0 / counts.clamp_min(1.0)
        ce_weights = inv / inv.mean()
        self.register_buffer("ce_weights", ce_weights)
        self.ldam = LDAMLoss(cls_num_list, max_m=ldam_max_m, scale=ldam_scale, class_balance_beta=class_balance_beta)
        self.focal = FocalLoss(gamma=focal_gamma)

    def forward(self, outputs: dict, targets: torch.Tensor) -> MACELossOutput:
        expert_logits = outputs["expert_logits"]
        ce = F.cross_entropy(expert_logits[0], targets)
        ldam = self.ldam(expert_logits[1], targets)
        focal = self.focal(expert_logits[2], targets)
        specialization = ce + ldam + focal
        aggregation = F.cross_entropy(outputs["logits"], targets, weight=self.ce_weights.to(targets.device))
        total = specialization + self.lambda_agg * aggregation
        return MACELossOutput(total, specialization, aggregation, ce, ldam, focal)
