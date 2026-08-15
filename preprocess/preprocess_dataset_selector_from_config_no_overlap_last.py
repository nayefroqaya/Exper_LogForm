#!/usr/bin/env python3
"""
Preprocess source and target datasets independently for supervised
cross-domain LogFormer experiments.

Example:
    Source = BGL
    Target = HDFS

Outputs:
    BGL_training_block_w{window}.npz
    BGL_validation_block_w{window}.npz
    BGL_testing_block_w{window}.npz

    HDFS_training_block_w{window}.npz
    HDFS_validation_block_w{window}.npz
    HDFS_testing_block_w{window}.npz

IMPORTANT:
- Source and target are NOT concatenated.
- No target fraction is sampled here.
- The complete target TRAIN split (normal + anomaly) is saved.
- The target fraction is selected later inside tune_transformer_pkl_ready_last.py.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip install pyyaml") from exc


SEQUENCE_COL = "__sequence_id__"


# ============================================================
# Configuration helpers
# ============================================================

def load_config(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("YAML config must contain a dictionary.")

    return cfg


def get_source_name(cfg: Dict[str, Any]) -> str:
    # New key
    if cfg.get("source_log_name"):
        return str(cfg["source_log_name"])

    # Compatible with your existing YAML
    names = cfg.get("source_dataset_names")
    if isinstance(names, list) and len(names) == 1:
        return str(names[0])

    if isinstance(names, list) and len(names) > 1:
        raise ValueError(
            "This supervised pretrain->tune pipeline expects one source dataset. "
            f"Found: {names}"
        )

    raise ValueError(
        "Specify either source_log_name: BGL or source_dataset_names: [BGL]"
    )


def get_target_name(cfg: Dict[str, Any]) -> str:
    if cfg.get("target_log_name"):
        return str(cfg["target_log_name"])

    if cfg.get("target_dataset_name"):
        return str(cfg["target_dataset_name"])

    raise ValueError(
        "Specify either target_log_name: HDFS or target_dataset_name: HDFS"
    )


def get_output_dir(cfg: Dict[str, Any]) -> str:
    return str(
        cfg.get(
            "preprocessed_dir",
            cfg.get("output_dir", "preprocess/preprocessed_data"),
        )
    )


def get_dataset_cfg(cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    datasets = cfg.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Missing datasets: section in YAML.")

    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}"
        )

    ds_cfg = datasets[dataset_name]

    for key in ["train_pkl", "val_pkl", "test_pkl"]:
        if key not in ds_cfg or ds_cfg[key] is None:
            raise ValueError(f"Dataset '{dataset_name}' missing '{key}'.")

    return ds_cfg


# ============================================================
# Input normalization
# ============================================================

def read_pkl(path: str, dataset_name: str, split_name: str) -> pd.DataFrame:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"{dataset_name} {split_name} PKL not found: {path_obj}"
        )

    print(f"Reading {dataset_name} {split_name}: {path_obj}")

    df = pd.read_pickle(path_obj).copy()

    df["__dataset_name__"] = dataset_name
    df["__split_name__"] = split_name

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "processed_EventTemplate" in df.columns and "EventTemplate" not in df.columns:
        df = df.rename(columns={"processed_EventTemplate": "EventTemplate"})

    # Preserve your previous behavior:
    # Original_Label is treated as the ground-truth label when it exists.
    if "Original_Label" in df.columns:
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])
        df = df.rename(columns={"Original_Label": "Label"})

    for col in ["EventTemplate", "Label"]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}'. "
                f"Available columns: {list(df.columns)}"
            )

    return df


def is_normal_label(value) -> bool:
    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "-", "normal", "0", "false", "benign"
    }


# ============================================================
# Sequence / block handling
# ============================================================

def require_block_col(
    df: pd.DataFrame,
    block_col: str,
    dataset_name: str,
    split_name: str,
):
    if block_col not in df.columns:
        raise ValueError(
            f"{dataset_name} {split_name}: '{block_col}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )


def check_split_overlap(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    block_col: str,
    strict: bool,
):
    """
    Check overlap using the ORIGINAL block IDs before prefixes are added.
    This is useful for HDFS because the same BlockId must not be present
    in train and test.
    """
    train_ids = set(train_df[block_col].astype(str).unique())
    val_ids = set(val_df[block_col].astype(str).unique())
    test_ids = set(test_df[block_col].astype(str).unique())

    overlaps = {
        "train/validation": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "validation/test": val_ids & test_ids,
    }

    print(f"\n{dataset_name} original {block_col} overlap check:")

    any_overlap = False
    for pair, ids in overlaps.items():
        print(f"  {pair}: {len(ids)}")
        any_overlap = any_overlap or bool(ids)

    if any_overlap and strict:
        raise ValueError(
            f"{dataset_name}: overlapping original block IDs were found across "
            "train/validation/test. Set strict_split_overlap_check: false only "
            "if these IDs are known to restart independently per split."
        )


def assign_existing_blocks(
    df: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    block_col: str,
) -> pd.DataFrame:
    """
    Keep Node_block_id unchanged and create a safe internal ID.

    Example:
        original Node_block_id: 123
        internal ID: BGL__train__123
    """
    require_block_col(df, block_col, dataset_name, split_name)

    out = df.copy()
    out[SEQUENCE_COL] = (
        str(dataset_name)
        + "__"
        + str(split_name)
        + "__"
        + out[block_col].astype(str)
    )
    return out


def assign_fixed_nonoverlap_windows(
    df: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    window_size: int,
) -> pd.DataFrame:
    """
    Optional mode for BGL when you want windows to be recreated directly
    from row order.

    Rows are NOT shuffled.
    The final incomplete window is dropped.
    """
    out = df.copy().reset_index(drop=True)

    complete_rows = (len(out) // window_size) * window_size
    dropped = len(out) - complete_rows

    if complete_rows == 0:
        raise ValueError(
            f"{dataset_name} {split_name}: not enough rows for "
            f"window_size={window_size}"
        )

    if dropped > 0:
        print(
            f"{dataset_name} {split_name}: dropping {dropped} trailing rows "
            "that do not form a complete window."
        )

    out = out.iloc[:complete_rows].copy()
    sequence_numbers = np.arange(complete_rows) // window_size

    out[SEQUENCE_COL] = [
        f"{dataset_name}__{split_name}__window_{i}"
        for i in sequence_numbers
    ]

    return out


def prepare_dataset(
    cfg: Dict[str, Any],
    dataset_name: str,
    window_size: int,
    default_block_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ds_cfg = get_dataset_cfg(cfg, dataset_name)

    # Default: preserve the block IDs already present in your PKLs.
    sequence_mode = ds_cfg.get(
        "sequence_mode",
        cfg.get("sequence_mode", "existing_blocks"),
    )

    block_length_mode = ds_cfg.get(
        "block_length_mode",
        cfg.get("block_length_mode", "keep_variable"),
    )

    block_col = ds_cfg.get(
        "block_col",
        cfg.get("block_col", default_block_col),
    )

    global_strict = bool(cfg.get("strict_split_overlap_check", False))
    strict_overlap = bool(
        ds_cfg.get("strict_split_overlap_check", global_strict)
    )

    if sequence_mode not in {"existing_blocks", "fixed_nonoverlap_rows"}:
        raise ValueError(
            "sequence_mode must be existing_blocks or fixed_nonoverlap_rows"
        )

    if block_length_mode not in {"exact", "keep_variable", "pad_truncate"}:
        raise ValueError(
            "block_length_mode must be exact, keep_variable, or pad_truncate"
        )

    print("\n============================================================")
    print(f"Dataset: {dataset_name}")
    print(f"sequence_mode: {sequence_mode}")
    print(f"block_length_mode: {block_length_mode}")
    print(f"window_size: {window_size}")
    print("============================================================")

    train_df = normalize_columns(
        read_pkl(ds_cfg["train_pkl"], dataset_name, "train")
    )
    val_df = normalize_columns(
        read_pkl(ds_cfg["val_pkl"], dataset_name, "validation")
    )
    test_df = normalize_columns(
        read_pkl(ds_cfg["test_pkl"], dataset_name, "testing")
    )

    if sequence_mode == "existing_blocks":
        require_block_col(train_df, block_col, dataset_name, "train")
        require_block_col(val_df, block_col, dataset_name, "validation")
        require_block_col(test_df, block_col, dataset_name, "testing")

        check_split_overlap(
            dataset_name,
            train_df,
            val_df,
            test_df,
            block_col,
            strict_overlap,
        )

        train_df = assign_existing_blocks(
            train_df, dataset_name, "train", block_col
        )
        val_df = assign_existing_blocks(
            val_df, dataset_name, "validation", block_col
        )
        test_df = assign_existing_blocks(
            test_df, dataset_name, "testing", block_col
        )

    else:
        train_df = assign_fixed_nonoverlap_windows(
            train_df, dataset_name, "train", window_size
        )
        val_df = assign_fixed_nonoverlap_windows(
            val_df, dataset_name, "validation", window_size
        )
        test_df = assign_fixed_nonoverlap_windows(
            test_df, dataset_name, "testing", window_size
        )

        # fixed windows are already exact by construction
        block_length_mode = "exact"

    return train_df, val_df, test_df, block_length_mode


# ============================================================
# Embeddings
# ============================================================

def add_or_reuse_vectors(
    dfs: List[pd.DataFrame],
    model_name: str,
    batch_size: int,
    device: str,
) -> List[pd.DataFrame]:
    if all("Vector" in df.columns for df in dfs):
        print("Vector exists in all splits. Reusing existing vectors.")
        return dfs

    print(f"\nLoading embedding model: {model_name}")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    templates = pd.concat(
        [df["EventTemplate"] for df in dfs],
        ignore_index=True,
    )
    templates = templates.dropna().astype(str).unique()

    print(f"Encoding {len(templates)} unique templates")

    embeddings = model.encode(
        templates.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    lookup = dict(zip(templates, embeddings))

    outputs = []

    for df in dfs:
        out = df.copy()
        out["Vector"] = out["EventTemplate"].astype(str).map(lookup)

        if out["Vector"].isna().any():
            raise ValueError("Some EventTemplate values could not be embedded.")

        outputs.append(out)

    return outputs


# ============================================================
# NPZ conversion
# ============================================================

def make_npz(
    df: pd.DataFrame,
    dataset_name: str,
    mode: str,
    output_dir: str,
    window_size: int,
    block_length_mode: str,
):
    """
    One NPZ entry = one complete sequence/block.

    Label:
        [1, 0] -> normal
        [0, 1] -> anomalous

    A sequence is anomalous if at least one row in the sequence is anomalous.
    """
    x_data = []
    y_data = []

    normal_count = 0
    anomaly_count = 0
    skipped = 0
    padded = 0
    truncated = 0

    for _, block_df in tqdm(
        df.groupby(SEQUENCE_COL, sort=False),
        desc=f"{dataset_name} {mode}",
    ):
        vectors = np.asarray(
            block_df["Vector"].tolist(),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or len(vectors) == 0:
            skipped += 1
            continue

        if block_length_mode == "exact":
            if len(vectors) != window_size:
                skipped += 1
                continue
            x_item = vectors

        elif block_length_mode == "keep_variable":
            x_item = vectors

        else:  # pad_truncate
            if len(vectors) >= window_size:
                x_item = vectors[:window_size]
                if len(vectors) > window_size:
                    truncated += 1
            else:
                pad_len = window_size - len(vectors)
                pad = np.zeros(
                    (pad_len, vectors.shape[1]),
                    dtype=np.float32,
                )
                x_item = np.vstack([vectors, pad])
                padded += 1

        labels = block_df["Label"].tolist()

        if all(is_normal_label(v) for v in labels):
            y = [1, 0]
            normal_count += 1
        else:
            y = [0, 1]
            anomaly_count += 1

        x_data.append(x_item)
        y_data.append(y)

    if block_length_mode == "keep_variable":
        x_data = np.array(x_data, dtype=object)
    else:
        x_data = np.asarray(x_data, dtype=np.float32)

    y_data = np.asarray(y_data, dtype=np.float32)

    if len(y_data) == 0:
        raise ValueError(
            f"{dataset_name} {mode}: generated dataset is empty. "
            "Check block_length_mode and window_size."
        )

    os.makedirs(output_dir, exist_ok=True)

    path = Path(output_dir) / (
        f"{dataset_name}_{mode}_block_w{window_size}.npz"
    )

    np.savez(path, x=x_data, y=y_data)

    print(f"\nSaved: {path}")
    print(f"  sequences: {len(y_data)}")
    print(f"  normal: {normal_count}")
    print(f"  anomaly: {anomaly_count}")
    print(f"  skipped: {skipped}")
    print(f"  padded: {padded}")
    print(f"  truncated: {truncated}")


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

    source_name = get_source_name(cfg)
    target_name = get_target_name(cfg)

    if source_name == target_name:
        raise ValueError("Source and target datasets must be different.")

    window_size = int(cfg.get("window_size", 120))
    output_dir = get_output_dir(cfg)
    default_block_col = cfg.get("block_col", "Node_block_id")

    embedding_model = cfg.get(
        "embedding_model",
        "distilbert-base-nli-mean-tokens",
    )
    embed_batch_size = int(cfg.get("embed_batch_size", 128))

    device_value = cfg.get("device")
    if device_value is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device_value)

    print("\n============================================================")
    print("SUPERVISED CROSS-DOMAIN PREPROCESSING")
    print("============================================================")
    print(f"Source: {source_name}")
    print(f"Target: {target_name}")
    print(f"Window size: {window_size}")
    print(f"Output directory: {output_dir}")
    print("NO source/target concatenation.")
    print("NO target fraction is selected during preprocessing.")
    print("============================================================")

    source_train, source_val, source_test, source_length_mode = prepare_dataset(
        cfg,
        source_name,
        window_size,
        default_block_col,
    )

    target_train, target_val, target_test, target_length_mode = prepare_dataset(
        cfg,
        target_name,
        window_size,
        default_block_col,
    )

    (
        source_train,
        source_val,
        source_test,
        target_train,
        target_val,
        target_test,
    ) = add_or_reuse_vectors(
        [
            source_train,
            source_val,
            source_test,
            target_train,
            target_val,
            target_test,
        ],
        embedding_model,
        embed_batch_size,
        device,
    )

    # SOURCE
    make_npz(
        source_train,
        source_name,
        "training",
        output_dir,
        window_size,
        source_length_mode,
    )
    make_npz(
        source_val,
        source_name,
        "validation",
        output_dir,
        window_size,
        source_length_mode,
    )
    make_npz(
        source_test,
        source_name,
        "testing",
        output_dir,
        window_size,
        source_length_mode,
    )

    # TARGET -- complete training set, both classes
    make_npz(
        target_train,
        target_name,
        "training",
        output_dir,
        window_size,
        target_length_mode,
    )
    make_npz(
        target_val,
        target_name,
        "validation",
        output_dir,
        window_size,
        target_length_mode,
    )
    make_npz(
        target_test,
        target_name,
        "testing",
        output_dir,
        window_size,
        target_length_mode,
    )

    print("\n============================================================")
    print("FINISHED")
    print("============================================================")
    print("Created independent source and target NPZ files.")
    print("The target TRAIN NPZ contains normal + anomalous blocks.")
    print(
        "The supervised fraction will be sampled later by "
        "tune_transformer_pkl_ready_last.py."
    )
    print("============================================================")


if __name__ == "__main__":
    main()