import argparse
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import nets
from nets import model_entrypoint


@dataclass
class FoldData:
    train: List[Data]
    val: List[Data]
    test: List[Data]
    mean: float
    std: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def structure_to_data(structure, target: float) -> Data:
    z = torch.tensor(structure.atomic_numbers, dtype=torch.long)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)
    y = torch.tensor([float(target)], dtype=torch.float)
    return Data(z=z, pos=pos, y=y)


def build_fold_data(task, fold: int, val_ratio: float, seed: int) -> FoldData:
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    train_val_targets = np.asarray(train_outputs, dtype=np.float32)
    mean = float(train_val_targets.mean())
    std = float(train_val_targets.std())
    if std < 1e-12:
        std = 1.0

    train_val_graphs = [
        structure_to_data(structure, target)
        for structure, target in zip(train_inputs, train_val_targets)
    ]

    num_train_val = len(train_val_graphs)
    val_size = max(1, int(num_train_val * val_ratio))
    val_size = min(val_size, num_train_val - 1)
    generator = torch.Generator().manual_seed(seed + int(fold))
    permutation = torch.randperm(num_train_val, generator=generator).tolist()
    val_indices = set(permutation[:val_size])

    train_graphs = [g for i, g in enumerate(train_val_graphs) if i not in val_indices]
    val_graphs = [g for i, g in enumerate(train_val_graphs) if i in val_indices]

    test_graphs = [
        structure_to_data(structure, target)
        for structure, target in zip(test_inputs, np.asarray(test_outputs, dtype=np.float32))
    ]

    return FoldData(train=train_graphs, val=val_graphs, test=test_graphs, mean=mean, std=std)


def train_epoch(model, loader, optimizer, device, mean: float, std: float) -> float:
    model.train()
    total_loss = 0.0
    total_size = 0

    for batch in loader:
        batch = batch.to(device)
        pred_energy, _ = model(node_atom=batch.z, pos=batch.pos, batch=batch.batch)
        pred_energy = pred_energy.view(-1)

        target = (batch.y.view(-1) - mean) / std
        loss = torch.nn.functional.l1_loss(pred_energy, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = target.numel()
        total_loss += loss.item() * bs
        total_size += bs

    return total_loss / max(total_size, 1)


@torch.no_grad()
def evaluate_mae(model, loader, device, mean: float, std: float) -> float:
    model.eval()
    total_mae = 0.0
    total_size = 0

    for batch in loader:
        batch = batch.to(device)
        pred_energy, _ = model(node_atom=batch.z, pos=batch.pos, batch=batch.batch)
        pred = pred_energy.view(-1) * std + mean
        target = batch.y.view(-1)

        mae = torch.abs(pred - target)
        total_mae += mae.sum().item()
        total_size += target.numel()

    return total_mae / max(total_size, 1)


def run_fold(args, task, fold: int, device: torch.device) -> float:
    fold_data = build_fold_data(task, fold, val_ratio=args.val_ratio, seed=args.seed)

    train_loader = DataLoader(
        fold_data.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        fold_data.val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        fold_data.test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    create_model = model_entrypoint(args.model_name)
    model = create_model(
        irreps_in=args.input_irreps,
        radius=args.radius,
        num_basis=args.num_basis,
        task_mean=fold_data.mean,
        task_std=fold_data.std,
        atomref=None,
        drop_path=args.drop_path,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_test = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            fold_data.mean,
            fold_data.std,
        )
        val_mae = evaluate_mae(
            model,
            val_loader,
            device,
            fold_data.mean,
            fold_data.std,
        )
        test_mae = evaluate_mae(
            model,
            test_loader,
            device,
            fold_data.mean,
            fold_data.std,
        )
        if val_mae <= best_val:
            best_val = val_mae
            best_test = test_mae

        print(
            f"[fold {fold:02d}] epoch {epoch:03d}/{args.epochs} "
            f"train_l1={train_loss:.6f} val_mae={val_mae:.6f} test_mae={test_mae:.6f} "
            f"best_val={best_val:.6f} best_test={best_test:.6f}"
        )

    return best_test


def parse_args():
    parser = argparse.ArgumentParser("Train Equiformer on Matbench JDFT2D")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # Keep model defaults aligned with repository defaults.
    parser.add_argument("--model-name", type=str, default="graph_attention_transformer_nonlinear_l2_md17")
    parser.add_argument("--input-irreps", type=str, default="64x0e")
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--num-basis", type=int, default=128)
    parser.add_argument("--drop-path", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    parser.add_argument("--max-folds", type=int, default=None, help="Only run first N folds for quick debug")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    from matbench.bench import MatbenchBenchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mb = MatbenchBenchmark(
        autoload=False,
        subset=["matbench_jdft2d"],
    )

    for task in mb.tasks:
        task.load()
        print(f"Loaded task: {task.dataset_name}, folds={len(task.folds)}")

        folds = task.folds
        if args.max_folds is not None:
            folds = folds[: args.max_folds]

        fold_test_scores = []
        for fold in folds:
            best_test = run_fold(args, task, fold, device)
            fold_test_scores.append(best_test)

        mean_best = float(np.mean(fold_test_scores)) if fold_test_scores else float("nan")
        print(f"Task {task.dataset_name}: mean test MAE over {len(fold_test_scores)} folds = {mean_best:.6f}")


if __name__ == "__main__":
    main()
