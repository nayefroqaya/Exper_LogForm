#!/usr/bin/env python3
"""
Supervised target-domain adaptation for LogFormer.

For BGL -> HDFS:
    1. Load the best BGL source checkpoint.
    2. Load COMPLETE HDFS training NPZ.
    3. Select X% of HDFS TRAIN using STRATIFIED sampling:
           X% of normal HDFS blocks
         + X% of anomalous HDFS blocks
       Thus target adaptation is supervised.
    4. Tune the adapter on this labeled target fraction.
    5. Use HDFS validation to select the best checkpoint.
    6. Evaluate HDFS test exactly once at the end.

IMPORTANT:
The target fraction is selected at NPZ-entry level.
Because each NPZ entry is already one complete block/sequence,
sampling never cuts a Node_block_id block into pieces.
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


def get_target_name(cfg):
    if cfg.get("target_log_name"):
        return str(cfg["target_log_name"])

    if cfg.get("target_dataset_name"):
        return str(cfg["target_dataset_name"])

    raise ValueError(
        "Specify target_log_name: HDFS or target_dataset_name: HDFS"
    )


def get_preprocessed_dir(cfg):
    return str(
        cfg.get(
            "preprocessed_dir",
            cfg.get("output_dir", "preprocess/preprocessed_data"),
        )
    )


def normalize_state_dict_keys(state_dict):
    """
    Accept checkpoints saved with or without nn.DataParallel.
    """
    clean = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        clean[key] = value

    return clean


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


def select_supervised_target_fraction(
    x,
    y,
    fraction,
    seed,
):
    """
    Stratified supervised target sampling.

    y encoding:
        [1, 0] = normal   -> argmax 0
        [0, 1] = anomaly  -> argmax 1

    For fraction=0.20:
        select 20% of normal target training blocks
        AND
        select 20% of anomalous target training blocks.

    This approximately preserves the original HDFS class ratio.
    """

    if not (0 < fraction <= 1):
        raise ValueError(
            "target_train_fraction must be > 0 and <= 1."
        )

    labels = np.argmax(y, axis=1)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    if len(normal_idx) == 0:
        raise ValueError(
            "Target training data contains no normal blocks."
        )

    if len(anomaly_idx) == 0:
        raise ValueError(
            "Target training data contains no anomalous blocks. "
            "Supervised LogFormer target adaptation requires both classes."
        )

    print("\nTarget TRAIN before sampling:")
    print(f"  total:   {len(y)}")
    print(f"  normal:  {len(normal_idx)}")
    print(f"  anomaly: {len(anomaly_idx)}")

    if fraction == 1.0:
        selected = np.arange(len(y))
    else:
        rng = np.random.default_rng(seed)

        n_normal = max(
            1,
            int(round(len(normal_idx) * fraction)),
        )
        n_anomaly = max(
            1,
            int(round(len(anomaly_idx) * fraction)),
        )

        n_normal = min(n_normal, len(normal_idx))
        n_anomaly = min(n_anomaly, len(anomaly_idx))

        selected_normal = rng.choice(
            normal_idx,
            size=n_normal,
            replace=False,
        )

        selected_anomaly = rng.choice(
            anomaly_idx,
            size=n_anomaly,
            replace=False,
        )

        selected = np.concatenate(
            [selected_normal, selected_anomaly]
        )

        rng.shuffle(selected)

    x_selected = x[selected]
    y_selected = y[selected]

    selected_labels = np.argmax(
        y_selected,
        axis=1,
    )

    print("\nSUPERVISED target fraction:")
    print(f"  fraction: {fraction:.4f} ({fraction * 100:.2f}%)")
    print(f"  total:    {len(y_selected)}")
    print(f"  normal:   {int(np.sum(selected_labels == 0))}")
    print(f"  anomaly:  {int(np.sum(selected_labels == 1))}")

    return x_selected, y_selected


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
    target_name = get_target_name(cfg)
    preprocessed_dir = get_preprocessed_dir(cfg)

    window_size = int(cfg.get("window_size", 120))
    seed = int(cfg.get("seed", 123))

    embedding_dim = int(cfg.get("embedding_dim", 768))
    num_layers = int(cfg.get("num_layers", 1))
    adapter_size = int(cfg.get("adapter_size", 64))
    nhead = int(cfg.get("nhead", 8))
    dropout = float(cfg.get("dropout", 0.1))
    feedforward_multiplier = int(cfg.get("feedforward_multiplier", 4))

    source_mode = str(cfg.get("source_mode", "classifier"))
    source_lr = float(cfg.get("source_lr", 1e-5))

    tune_mode = str(cfg.get("tune_mode", "adapter"))
    batch_size = int(cfg.get("target_batch_size", 64))
    epochs = int(cfg.get("target_epochs", 20))
    lr = float(cfg.get("target_lr", 1e-5))

    # Preferred new key.
    # For compatibility, your old target_normal_fraction is accepted,
    # but it is now interpreted as a fraction of BOTH classes.
    if cfg.get("target_train_fraction") is not None:
        target_fraction = float(cfg["target_train_fraction"])
    elif cfg.get("target_normal_fraction") is not None:
        target_fraction = float(cfg["target_normal_fraction"])
        print(
            "WARNING: using legacy YAML key 'target_normal_fraction'. "
            "In this supervised script it means the fraction sampled "
            "from BOTH normal and anomalous target training blocks. "
            "Rename it to 'target_train_fraction' for clarity."
        )
    else:
        target_fraction = 1.0

    sampling_seed = int(
        cfg.get("target_sampling_seed", seed)
    )

    reinitialize_classifier = bool(
        cfg.get("reinitialize_classifier", True)
    )

    result_dir = str(cfg.get("result_dir", "result"))
    checkpoint_dir = str(cfg.get("checkpoint_dir", "checkpoints"))

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    source_suffix = (
        f"{source_name}_{source_mode}_{num_layers}_{adapter_size}_{source_lr}"
    )

    source_checkpoint = cfg.get(
        "source_best_checkpoint",
        os.path.join(
            checkpoint_dir,
            f"train_{source_suffix}-best.pt",
        ),
    )

    target_suffix = (
        f"{target_name}_from_{source_name}_"
        f"{tune_mode}_{num_layers}_{adapter_size}_{lr}_{epochs}"
    )

    result_file = os.path.join(
        result_dir,
        f"tune_{target_suffix}.txt",
    )

    best_path = cfg.get(
        "target_best_checkpoint",
        os.path.join(
            checkpoint_dir,
            f"tune_{target_suffix}-best.pt",
        ),
    )

    latest_path = cfg.get(
        "target_latest_checkpoint",
        os.path.join(
            checkpoint_dir,
            f"tune_{target_suffix}-latest.pt",
        ),
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
    # Load TARGET train/val/test
    # -------------------------
    train_path = os.path.join(
        preprocessed_dir,
        f"{target_name}_training_block_w{window_size}.npz",
    )
    val_path = os.path.join(
        preprocessed_dir,
        f"{target_name}_validation_block_w{window_size}.npz",
    )
    test_path = os.path.join(
        preprocessed_dir,
        f"{target_name}_testing_block_w{window_size}.npz",
    )

    train_npz = np.load(train_path, allow_pickle=True)
    val_npz = np.load(val_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)

    x_train_full, y_train_full = train_npz["x"], train_npz["y"]
    x_val, y_val = val_npz["x"], val_npz["y"]
    x_test, y_test = test_npz["x"], test_npz["y"]

    del train_npz, val_npz, test_npz

    if len(x_train_full) == 0:
        raise ValueError("Target training NPZ is empty.")

    if len(x_val) == 0:
        raise ValueError("Target validation NPZ is empty.")

    if len(x_test) == 0:
        raise ValueError("Target testing NPZ is empty.")

    # -------------------------
    # SUPERVISED target sampling
    # -------------------------
    x_train, y_train = select_supervised_target_fraction(
        x_train_full,
        y_train_full,
        target_fraction,
        sampling_seed,
    )

    del x_train_full, y_train_full

    def print_distribution(name, y):
        labels = np.argmax(y, axis=1)
        print(
            f"{name}: total={len(y)}, "
            f"normal={int(np.sum(labels == 0))}, "
            f"anomaly={int(np.sum(labels == 1))}"
        )

    print("\n============================================================")
    print("SUPERVISED TARGET ADAPTATION")
    print("============================================================")
    print("Source checkpoint:", source_checkpoint)
    print("Source:", source_name)
    print("Target:", target_name)
    print(f"Target fraction: {target_fraction * 100:.2f}%")
    print_distribution("Selected target train", y_train)
    print_distribution("Target validation", y_val)
    print_distribution("Target test", y_test)
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
    # Build target adapter model
    # -------------------------
    model = Model(
        mode="adapter",
        num_layers=num_layers,
        adapter_size=adapter_size,
        dim=embedding_dim,
        window_size=window_size,
        nhead=nhead,
        dim_feedforward=feedforward_multiplier * embedding_dim,
        dropout=dropout,
    )

    # -------------------------
    # Load SOURCE checkpoint
    # -------------------------
    if not os.path.exists(source_checkpoint):
        raise FileNotFoundError(
            f"Source checkpoint not found: {source_checkpoint}"
        )

    checkpoint = torch.load(
        source_checkpoint,
        map_location="cpu",
    )

    source_state = normalize_state_dict_keys(
        checkpoint["net"]
    )

    # Follow your original tuning behavior:
    # optionally start the target classifier head fresh.
    if reinitialize_classifier:
        source_state.pop("fc1.weight", None)
        source_state.pop("fc1.bias", None)
        print("Target classifier head will be reinitialized.")

    load_result = model.load_state_dict(
        source_state,
        strict=False,
    )

    print("Source checkpoint load result:")
    print(load_result)

    # -------------------------
    # Choose what is trainable
    # -------------------------
    if tune_mode == "adapter":
        model.train_adapter()

    elif tune_mode == "classifier":
        model.train_classifier()

    elif tune_mode == "tuning":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(
            "tune_mode must be adapter, classifier, or tuning"
        )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable_params:
        raise ValueError(
            "No trainable parameters. Check Model.train_adapter() "
            "or tune_mode."
        )

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_count = sum(
        p.numel() for p in trainable_params
    )

    print(
        f"Trainable parameters: {trainable_count}/{total_params} "
        f"({100.0 * trainable_count / total_params:.2f}%)"
    )

    optimizer = optim.Adam(
        trainable_params,
        lr=lr,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.BCEWithLogitsLoss()

    # -------------------------
    # Target tuning
    # -------------------------
    best_val_f1 = -1.0
    total_start = time.time()

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("SUPERVISED CROSS-DOMAIN TARGET TUNING\n")
        f.write(f"source={source_name}\n")
        f.write(f"target={target_name}\n")
        f.write(f"target_fraction={target_fraction}\n")
        f.write("target_classes=normal+anomaly\n")
        f.write(f"tune_mode={tune_mode}\n")
        f.write(f"window_size={window_size}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"lr={lr}\n")
        f.write(f"selected_target_train={len(y_train)}\n\n")

    for epoch in range(epochs):
        model.train()
        losses = []

        for x, y in tqdm(
            train_loader,
            desc=f"Target epoch {epoch + 1}/{epochs}",
        ):
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                trainable_params,
                0.5,
            )

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        # Validation decides best checkpoint.
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

        target_checkpoint = {
            "net": get_model_state(model),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_f1": val_f1,
            "source_dataset": source_name,
            "target_dataset": target_name,
            "target_fraction": target_fraction,
            "target_classes": "normal+anomaly",
            "window_size": window_size,
        }

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                target_checkpoint,
                best_path,
            )
            print("Saved best target checkpoint:", best_path)

        torch.save(
            target_checkpoint,
            latest_path,
        )

    # -------------------------
    # Final HDFS test ONCE
    # -------------------------
    best_checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(
            best_checkpoint["net"]
        )
    else:
        model.load_state_dict(
            best_checkpoint["net"]
        )

    test_p, test_r, test_f1, test_cm, test_time = evaluate(
        model,
        test_loader,
        device,
    )

    total_time = time.time() - total_start

    print("\n============================================================")
    print("FINAL TARGET TEST")
    print("============================================================")
    print("Source:", source_name)
    print("Target:", target_name)
    print(f"Target supervised fraction: {target_fraction * 100:.2f}%")
    print(f"Number of testing data: {len(y_test)}")
    print(f"Precision: {test_p:.4f}")
    print(f"Recall:    {test_r:.4f}")
    print(f"F1 score:  {test_f1:.4f}")
    print("Confusion matrix:")
    print(test_cm)
    print("Best checkpoint:", best_path)
    print("============================================================")

    with open(result_file, "a", encoding="utf-8") as f:
        f.write("\nFINAL TARGET TEST\n")
        f.write(f"number_testing_data={len(y_test)}\n")
        f.write(f"precision={test_p}\n")
        f.write(f"recall={test_r}\n")
        f.write(f"f1={test_f1}\n")
        f.write(f"confusion_matrix=\n{test_cm}\n")
        f.write(f"prediction_time={test_time}\n")
        f.write(f"total_time={total_time}\n")
        f.write(f"best_checkpoint={best_path}\n")


if __name__ == "__main__":
    main()