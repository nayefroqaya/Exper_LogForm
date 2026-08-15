#!/usr/bin/env python3
"""
Multi-source source-domain pretraining for LogFormer.

Supports:
    - 1, 2, or 3 source datasets
    - exactly 1 target dataset in YAML, but target data are NOT
      used by this script.

Example:
    source_dataset_names:
      - BGL
      - HDFS
    target_dataset_name: TH_1G

Training:
    BGL train + HDFS train -> one combined source training set.

Validation:
    Each source validation set is evaluated independently.
    The checkpoint is selected using MACRO mean F1 across source
    validation datasets, so a very large source does not dominate
    the checkpoint decision.

Target data:
    NOT USED HERE.
"""

import argparse
import os
import random
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from dataloader import DataGenerator
from model import Model


# ============================================================
# Configuration
# ============================================================

def load_config(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f) or {}


def get_source_names(cfg) -> List[str]:
    names = cfg.get(
        "source_dataset_names"
    )

    if not isinstance(names, list):
        raise ValueError(
            "source_dataset_names must be a YAML list."
        )

    names = [
        str(x)
        for x in names
    ]

    if not (1 <= len(names) <= 3):
        raise ValueError(
            "Use between 1 and 3 source datasets."
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate source datasets are not allowed."
        )

    target = str(
        cfg.get(
            "target_dataset_name",
            "",
        )
    )

    if target in names:
        raise ValueError(
            "Target dataset cannot also be a source."
        )

    return names


def get_preprocessed_dir(cfg):
    return str(
        cfg.get(
            "preprocessed_dir",
            cfg.get(
                "output_dir",
                "preprocess/preprocessed_data",
            ),
        )
    )


# ============================================================
# Array helpers
# ============================================================

def to_object_sequences(x):
    """
    Convert either:
        numeric array [N, W, 768]
    or:
        object array [N]
    into one object array where each item is a complete
    float32 sequence [L, 768].

    This lets fixed BGL sequences and variable HDFS sequences
    be safely combined without changing block boundaries.
    """
    out = np.empty(
        len(x),
        dtype=object,
    )

    for i in range(len(x)):
        seq = np.asarray(
            x[i],
            dtype=np.float32,
        )

        if (
            seq.ndim != 2
            or seq.shape[1] != 768
        ):
            raise ValueError(
                f"Invalid sequence shape at index {i}: "
                f"{seq.shape}. Expected [length, 768]."
            )

        out[i] = seq

    return out


def load_split(
    preprocessed_dir: str,
    dataset_name: str,
    split_name: str,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(
        preprocessed_dir,
        f"{dataset_name}_{split_name}_block_w{window_size}.npz",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing preprocessed file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    )

    x = data["x"]
    y = data["y"]

    del data

    if len(x) == 0:
        raise ValueError(
            f"{dataset_name} {split_name} is empty."
        )

    return x, y


def print_distribution(
    dataset_name,
    split_name,
    y,
):
    labels = np.argmax(
        y,
        axis=1,
    )

    print(
        f"{dataset_name} {split_name}: "
        f"total={len(y)}, "
        f"normal={int(np.sum(labels == 0))}, "
        f"anomaly={int(np.sum(labels == 1))}"
    )


# ============================================================
# Model helpers
# ============================================================

def get_model_state(model):
    if isinstance(
        model,
        nn.DataParallel,
    ):
        return model.module.state_dict()

    return model.state_dict()


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    pred_all = []
    true_all = []

    start = time.time()

    with torch.no_grad():
        for x, y in tqdm(
            loader,
            desc="Evaluation",
            leave=False,
        ):
            x = (
                x.to(device)
                .to(torch.float32)
            )
            y = (
                y.to(device)
                .to(torch.float32)
            )

            out = model(x)

            pred_all.append(
                out.cpu()
            )
            true_all.append(
                y.cpu()
            )

    if not pred_all:
        raise ValueError(
            "Evaluation loader is empty."
        )

    prediction_time = (
        time.time() - start
    )

    pred = torch.cat(
        pred_all,
        dim=0,
    ).numpy()

    true = torch.cat(
        true_all,
        dim=0,
    ).numpy()

    pred_labels = np.argmax(
        pred,
        axis=1,
    )

    true_labels = np.argmax(
        true,
        axis=1,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average="binary",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        true_labels,
        pred_labels,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "prediction_time": prediction_time,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="preprocess/config_cross_dataset_last.yml",
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    source_names = get_source_names(
        cfg
    )

    source_tag = "_".join(
        source_names
    )

    preprocessed_dir = (
        get_preprocessed_dir(cfg)
    )

    window_size = int(
        cfg.get("window_size", 120)
    )

    seed = int(
        cfg.get("seed", 123)
    )

    embedding_dim = int(
        cfg.get("embedding_dim", 768)
    )

    num_layers = int(
        cfg.get("num_layers", 1)
    )

    adapter_size = int(
        cfg.get("adapter_size", 64)
    )

    nhead = int(
        cfg.get("nhead", 8)
    )

    dropout = float(
        cfg.get("dropout", 0.1)
    )

    ff_multiplier = int(
        cfg.get(
            "feedforward_multiplier",
            4,
        )
    )

    source_mode = str(
        cfg.get(
            "source_mode",
            "classifier",
        )
    )

    batch_size = int(
        cfg.get(
            "source_batch_size",
            64,
        )
    )

    epochs = int(
        cfg.get(
            "source_epochs",
            5,
        )
    )

    lr = float(
        cfg.get(
            "source_lr",
            1e-5,
        )
    )

    result_dir = str(
        cfg.get(
            "result_dir",
            "result",
        )
    )

    checkpoint_dir = str(
        cfg.get(
            "checkpoint_dir",
            "checkpoints",
        )
    )

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    suffix = (
        f"{source_tag}_"
        f"{source_mode}_"
        f"{num_layers}_"
        f"{adapter_size}_"
        f"{lr}"
    )

    result_file = os.path.join(
        result_dir,
        f"train_{suffix}.txt",
    )

    default_best_path = os.path.join(
        checkpoint_dir,
        f"train_{suffix}-best.pt",
    )

    default_latest_path = os.path.join(
        checkpoint_dir,
        f"train_{suffix}-latest.pt",
    )

    # Optional explicit overrides.
    best_path = str(
        cfg.get(
            "source_best_checkpoint",
            default_best_path,
        )
    )

    latest_path = str(
        cfg.get(
            "source_latest_checkpoint",
            default_latest_path,
        )
    )

    Path(best_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(latest_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Reproducibility
    # -------------------------
    warnings.filterwarnings(
        "ignore"
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # ========================================================
    # Load ALL source datasets
    # ========================================================

    train_x_parts = []
    train_y_parts = []

    validation_sets = {}
    testing_sets = {}

    print("\n============================================================")
    print("MULTI-SOURCE PRETRAINING")
    print("============================================================")
    print("Sources:", source_names)
    print("Source tag:", source_tag)
    print("Target data used here: NONE")
    print("Window size:", window_size)
    print("============================================================")

    for source_name in source_names:
        x_train, y_train = load_split(
            preprocessed_dir,
            source_name,
            "training",
            window_size,
        )

        x_val, y_val = load_split(
            preprocessed_dir,
            source_name,
            "validation",
            window_size,
        )

        x_test, y_test = load_split(
            preprocessed_dir,
            source_name,
            "testing",
            window_size,
        )

        print_distribution(
            source_name,
            "train",
            y_train,
        )

        print_distribution(
            source_name,
            "validation",
            y_val,
        )

        print_distribution(
            source_name,
            "test",
            y_test,
        )

        train_x_parts.append(
            to_object_sequences(
                x_train
            )
        )

        train_y_parts.append(
            np.asarray(
                y_train,
                dtype=np.float32,
            )
        )

        validation_sets[source_name] = (
            to_object_sequences(x_val),
            np.asarray(
                y_val,
                dtype=np.float32,
            ),
        )

        testing_sets[source_name] = (
            to_object_sequences(x_test),
            np.asarray(
                y_test,
                dtype=np.float32,
            ),
        )

    # Combine full source training datasets.
    x_train_combined = np.concatenate(
        train_x_parts,
        axis=0,
    )

    y_train_combined = np.concatenate(
        train_y_parts,
        axis=0,
    )

    # Shuffle dataset order once here.
    # DataLoader will also shuffle every epoch.
    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        len(y_train_combined)
    )

    x_train_combined = (
        x_train_combined[permutation]
    )

    y_train_combined = (
        y_train_combined[permutation]
    )

    print_distribution(
        source_tag,
        "combined-train",
        y_train_combined,
    )

    # ========================================================
    # Data loaders
    # ========================================================

    train_loader = (
        torch.utils.data.DataLoader(
            DataGenerator(
                x_train_combined,
                y_train_combined,
                window_size,
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    )

    validation_loaders = {}

    for name, (x_val, y_val) in (
        validation_sets.items()
    ):
        validation_loaders[name] = (
            torch.utils.data.DataLoader(
                DataGenerator(
                    x_val,
                    y_val,
                    window_size,
                ),
                batch_size=batch_size,
                shuffle=False,
            )
        )

    testing_loaders = {}

    for name, (x_test, y_test) in (
        testing_sets.items()
    ):
        testing_loaders[name] = (
            torch.utils.data.DataLoader(
                DataGenerator(
                    x_test,
                    y_test,
                    window_size,
                ),
                batch_size=batch_size,
                shuffle=False,
            )
        )

    # ========================================================
    # Source model
    # ========================================================

    model = Model(
        mode=source_mode,
        num_layers=num_layers,
        adapter_size=adapter_size,
        dim=embedding_dim,
        window_size=window_size,
        nhead=nhead,
        dim_feedforward=(
            ff_multiplier
            * embedding_dim
        ),
        dropout=dropout,
    )

    model = model.to(
        device
    )

    if (
        torch.cuda.device_count()
        > 1
    ):
        print(
            "Using",
            torch.cuda.device_count(),
            "GPUs",
        )

        model = nn.DataParallel(
            model
        )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0,
    )

    scheduler = (
        optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(
                train_loader
            ),
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    # ========================================================
    # Train
    # ========================================================

    best_macro_val_f1 = -1.0

    total_start = time.time()

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "MULTI-SOURCE LOGFORMER PRETRAINING\n"
        )
        f.write(
            f"sources={source_names}\n"
        )
        f.write(
            f"source_tag={source_tag}\n"
        )
        f.write(
            f"window_size={window_size}\n"
        )
        f.write(
            f"epochs={epochs}\n"
        )
        f.write(
            f"lr={lr}\n"
        )
        f.write(
            f"batch_size={batch_size}\n"
        )
        f.write(
            f"combined_training_samples="
            f"{len(y_train_combined)}\n\n"
        )

    for epoch in range(epochs):
        model.train()

        losses = []

        for x, y in tqdm(
            train_loader,
            desc=(
                f"Source epoch "
                f"{epoch + 1}/{epochs}"
            ),
        ):
            x = (
                x.to(device)
                .to(torch.float32)
            )

            y = (
                y.to(device)
                .to(torch.float32)
            )

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(
                out,
                y,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5,
            )

            optimizer.step()
            scheduler.step()

            losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(losses)
        )

        # --------------------------------------------
        # Evaluate each SOURCE validation independently
        # --------------------------------------------

        source_val_results = {}

        source_val_f1s = []

        print(
            f"\nEpoch {epoch + 1} "
            "source validation:"
        )

        for source_name in source_names:
            result = evaluate(
                model,
                validation_loaders[
                    source_name
                ],
                device,
            )

            source_val_results[
                source_name
            ] = result

            source_val_f1s.append(
                result["f1"]
            )

            print(
                f"  {source_name}: "
                f"P={result['precision']:.4f}, "
                f"R={result['recall']:.4f}, "
                f"F1={result['f1']:.4f}"
            )

        # Macro across source datasets, NOT weighted by
        # validation-set size.
        macro_val_f1 = float(
            np.mean(
                source_val_f1s
            )
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.6f}, "
            f"macro-source-Val-F1="
            f"{macro_val_f1:.4f}"
        )

        with open(
            result_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                f"Epoch {epoch + 1}: "
                f"loss={train_loss:.6f}, "
                f"macro_val_f1="
                f"{macro_val_f1:.6f}\n"
            )

            for source_name in (
                source_names
            ):
                r = source_val_results[
                    source_name
                ]

                f.write(
                    f"  {source_name}: "
                    f"P={r['precision']:.6f}, "
                    f"R={r['recall']:.6f}, "
                    f"F1={r['f1']:.6f}\n"
                )

        checkpoint = {
            "net": get_model_state(
                model
            ),
            "optimizer": (
                optimizer.state_dict()
            ),
            "epoch": epoch,
            "macro_val_f1": (
                macro_val_f1
            ),
            "source_datasets": (
                source_names
            ),
            "source_tag": source_tag,
            "window_size": (
                window_size
            ),
        }

        if (
            macro_val_f1
            > best_macro_val_f1
        ):
            best_macro_val_f1 = (
                macro_val_f1
            )

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                "Saved best source "
                "checkpoint:",
                best_path,
            )

        torch.save(
            checkpoint,
            latest_path,
        )

    # ========================================================
    # Final source tests once
    # ========================================================

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    if isinstance(
        model,
        nn.DataParallel,
    ):
        model.module.load_state_dict(
            best_checkpoint["net"]
        )
    else:
        model.load_state_dict(
            best_checkpoint["net"]
        )

    print("\n============================================================")
    print("FINAL SOURCE TESTS")
    print("============================================================")

    source_test_f1s = []

    with open(
        result_file,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            "\nFINAL SOURCE TESTS\n"
        )

        for source_name in source_names:
            result = evaluate(
                model,
                testing_loaders[
                    source_name
                ],
                device,
            )

            source_test_f1s.append(
                result["f1"]
            )

            print(
                f"{source_name}: "
                f"P={result['precision']:.4f}, "
                f"R={result['recall']:.4f}, "
                f"F1={result['f1']:.4f}"
            )

            print(
                result[
                    "confusion_matrix"
                ]
            )

            f.write(
                f"{source_name}: "
                f"P={result['precision']}, "
                f"R={result['recall']}, "
                f"F1={result['f1']}\n"
            )

            f.write(
                f"{result['confusion_matrix']}\n"
            )

        macro_test_f1 = float(
            np.mean(
                source_test_f1s
            )
        )

        total_time = (
            time.time()
            - total_start
        )

        print(
            "Macro source test F1:",
            f"{macro_test_f1:.4f}",
        )

        print(
            "Best checkpoint:",
            best_path,
        )

        f.write(
            f"macro_source_test_f1="
            f"{macro_test_f1}\n"
        )

        f.write(
            f"total_time={total_time}\n"
        )

        f.write(
            f"best_checkpoint="
            f"{best_path}\n"
        )

    print("============================================================")


if __name__ == "__main__":
    main()