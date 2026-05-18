from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import JsonlGraphDataset, collate_graphs, move_batch
from .losses import MACELoss
from .metrics import binary_metrics
from .model import VulHME, VulHMEConfig


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate(model: VulHME, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(
                batch["node_features"],
                batch["edge_index"],
                batch["edge_type"],
                batch["batch"],
                batch["spatial_pos"],
                batch["edge_bias"],
            )
            all_logits.append(outputs["logits"])
            all_targets.append(batch["targets"])
    return binary_metrics(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0))


def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and config["train"].get("cuda", True) else "cpu")
    datasets = {
        split: JsonlGraphDataset(config["data"][split])
        for split in ["train", "valid", "test"]
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=(split == "train"),
            collate_fn=collate_graphs,
        )
        for split, dataset in datasets.items()
    }

    model_config = VulHMEConfig(**config["model"])
    model = VulHME(model_config).to(device)
    criterion = MACELoss(config["data"]["class_counts"], **config["mace"]).to(device)
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_valid_loss = float("inf")
    patience = 0

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            outputs = model(
                batch["node_features"],
                batch["edge_index"],
                batch["edge_type"],
                batch["batch"],
                batch["spatial_pos"],
                batch["edge_bias"],
            )
            loss_output = criterion(outputs, batch["targets"])
            optimizer.zero_grad()
            loss_output.total.backward()
            optimizer.step()
            total_loss += float(loss_output.total.detach().cpu())

        valid_metrics = evaluate(model, loaders["valid"], device)
        train_loss = total_loss / max(len(loaders["train"]), 1)
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"valid_f1={valid_metrics['f1']:.2f}"
        )

        valid_loss_proxy = -valid_metrics["f1"]
        if valid_loss_proxy < best_valid_loss:
            best_valid_loss = valid_loss_proxy
            patience = 0
            torch.save(model.state_dict(), output_dir / "vulhme_best.pt")
        else:
            patience += 1
            if patience >= config["train"]["early_stop_patience"]:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
