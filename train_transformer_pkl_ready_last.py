#!/usr/bin/env python3
"""
Source-domain pretraining for supervised cross-domain LogFormer.

For BGL -> HDFS:
    - trains ONLY on BGL training
    - selects the best source checkpoint using BGL validation
    - evaluates BGL test only once after training
    - does NOT use any HDFS data
"""

import argparse
import os
import random
import time
import warnings
from pathlib import Path

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
# Helpers
# ============================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_source_name(cfg):
    if cfg.get("source_log_name"):
        return str(cfg["source_log_name"])

    names = cfg.get("source_dataset_names")
    if isinstance(names, list) and len(names) == 1:
        return str(names[0])

    raise ValueError(
        "Specify source_log_name: BGL or source_dataset_names: [BGL]"
    )


def get_preprocessed_dir(cfg):
    return str(
        cfg.get(
            "preprocessed_dir",
            cfg.get("output_dir", "preprocess/preprocessed_data"),
        )
    )


def get_model_state(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def evaluate(model, loader, device):
    model.eval()

    pred_all = []
    true_all = []

    start = time.time()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluation"):
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)

            out = model(x)

            pred_all.append(out.cpu())
            true_all.append(y.cpu())

    if not pred_all:
        raise ValueError("Evaluation loader is empty.")

    prediction_time = time.time() - start

    pred = torch.cat(pred_all, dim=0).numpy()
    true = torch.cat(true_all, dim=0).numpy()

    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(true, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, pred_labels)

    return precision, recall, f1, cm, prediction_time


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_cross_dataset_last.yml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # -------------------------
    # Experiment configuration
    # -------------------------
    source_name = get_source_name(cfg)
    preprocessed_dir = get_preprocessed_dir(cfg)

    window_size = int(cfg.get("window_size", 120))
    seed = int(cfg.get("seed", 123))

    embedding_dim = int(cfg.get("embedding_dim", 768))
    num_layers = int(cfg.get("num_layers", 1))
    adapter_size = int(cfg.get("adapter_size", 64))
    nhead = int(cfg.get("nhead", 8))
    dropout = float(cfg.get("dropout", 0.1))
    feedforward_multiplier = int(cfg.get("feedforward_multiplier", 4))

    mode = str(cfg.get("source_mode", "classifier"))
    batch_size = int(cfg.get("source_batch_size", 64))
    epochs = int(cfg.get("source_epochs", 5))
    lr = float(cfg.get("source_lr", 1e-5))

    result_dir = str(cfg.get("result_dir", "result"))
    checkpoint_dir = str(cfg.get("checkpoint_dir", "checkpoints"))

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    suffix = (
        f"{source_name}_{mode}_{num_layers}_{adapter_size}_{lr}"
    )

    result_file = os.path.join(
        result_dir,
        f"train_{suffix}.txt",
    )

    best_path = cfg.get(
        "source_best_checkpoint",
        os.path.join(checkpoint_dir, f"train_{suffix}-best.pt"),
    )

    latest_path = cfg.get(
        "source_latest_checkpoint",
        os.path.join(checkpoint_dir, f"train_{suffix}-latest.pt"),
    )

    Path(best_path).parent.mkdir(parents=True, exist_ok=True)
    Path(latest_path).parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Reproducibility
    # -------------------------
    warnings.filterwarnings("ignore")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # -------------------------
    # Load SOURCE only
    # -------------------------
    train_path = os.path.join(
        preprocessed_dir,
        f"{source_name}_training_block_w{window_size}.npz",
    )
    val_path = os.path.join(
        preprocessed_dir,
        f"{source_name}_validation_block_w{window_size}.npz",
    )
    test_path = os.path.join(
        preprocessed_dir,
        f"{source_name}_testing_block_w{window_size}.npz",
    )

    train_npz = np.load(train_path, allow_pickle=True)
    val_npz = np.load(val_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)

    x_train, y_train = train_npz["x"], train_npz["y"]
    x_val, y_val = val_npz["x"], val_npz["y"]
    x_test, y_test = test_npz["x"], test_npz["y"]

    del train_npz, val_npz, test_npz

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Source train/validation/test NPZ must all be non-empty.")

    def print_distribution(name, y):
        labels = np.argmax(y, axis=1)
        print(
            f"{name}: total={len(y)}, "
            f"normal={int(np.sum(labels == 0))}, "
            f"anomaly={int(np.sum(labels == 1))}"
        )

    print("\n============================================================")
    print("SOURCE PRETRAINING")
    print("============================================================")
    print("Source:", source_name)
    print("Target data used here: NONE")
    print_distribution("Source train", y_train)
    print_distribution("Source validation", y_val)
    print_distribution("Source test", y_test)
    print("============================================================")

    # -------------------------
    # Data loaders
    # -------------------------
    train_loader = torch.utils.data.DataLoader(
        DataGenerator(x_train, y_train, window_size),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        DataGenerator(x_val, y_val, window_size),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        DataGenerator(x_test, y_test, window_size),
        batch_size=batch_size,
        shuffle=False,
    )

    # -------------------------
    # Model
    # -------------------------
    model = Model(
        mode=mode,
        num_layers=num_layers,
        adapter_size=adapter_size,
        dim=embedding_dim,
        window_size=window_size,
        nhead=nhead,
        dim_feedforward=feedforward_multiplier * embedding_dim,
        dropout=dropout,
    )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.BCEWithLogitsLoss()

    # -------------------------
    # Train source model
    # -------------------------
    best_val_f1 = -1.0
    total_start = time.time()

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("SOURCE PRETRAINING\n")
        f.write(f"source={source_name}\n")
        f.write(f"window_size={window_size}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"lr={lr}\n")
        f.write(f"batch_size={batch_size}\n\n")

    for epoch in range(epochs):
        model.train()
        losses = []

        for x, y in tqdm(
            train_loader,
            desc=f"Source epoch {epoch + 1}/{epochs}",
        ):
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5,
            )

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        val_p, val_r, val_f1, val_cm, _ = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.6f}, "
            f"Val P={val_p:.4f}, "
            f"Val R={val_r:.4f}, "
            f"Val F1={val_f1:.4f}"
        )

        with open(result_file, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch + 1}: "
                f"loss={train_loss:.6f}, "
                f"val_precision={val_p:.6f}, "
                f"val_recall={val_r:.6f}, "
                f"val_f1={val_f1:.6f}\n"
            )

        checkpoint = {
            "net": get_model_state(model),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_f1": val_f1,
            "source_dataset": source_name,
            "window_size": window_size,
        }

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(checkpoint, best_path)
            print("Saved best source checkpoint:", best_path)

        torch.save(checkpoint, latest_path)

    # -------------------------
    # Final source test ONCE
    # -------------------------
    best_checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_checkpoint["net"])
    else:
        model.load_state_dict(best_checkpoint["net"])

    test_p, test_r, test_f1, test_cm, test_time = evaluate(
        model,
        test_loader,
        device,
    )

    total_time = time.time() - total_start

    print("\n============================================================")
    print("FINAL SOURCE TEST")
    print("============================================================")
    print(f"Precision: {test_p:.4f}")
    print(f"Recall:    {test_r:.4f}")
    print(f"F1:        {test_f1:.4f}")
    print("Confusion matrix:")
    print(test_cm)
    print("Best checkpoint:", best_path)
    print("============================================================")

    with open(result_file, "a", encoding="utf-8") as f:
        f.write("\nFINAL SOURCE TEST\n")
        f.write(f"precision={test_p}\n")
        f.write(f"recall={test_r}\n")
        f.write(f"f1={test_f1}\n")
        f.write(f"confusion_matrix=\n{test_cm}\n")
        f.write(f"prediction_time={test_time}\n")
        f.write(f"total_time={total_time}\n")
        f.write(f"best_checkpoint={best_path}\n")


if __name__ == "__main__":
    main()