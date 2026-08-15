#!/usr/bin/env python3
"""
Multi-source preprocessing for LogFormer.

Supports:
    - 1, 2, or 3 source datasets
    - exactly 1 target dataset

IMPORTANT DESIGN:
    - Every dataset is preprocessed independently.
    - Source datasets are NOT concatenated here.
    - Target data are NOT added to source data here.
    - No target fraction is sampled here.
    - The complete target TRAIN split is saved.
    - The target fraction (normal + anomaly) is sampled later in
      tune_transformer_pkl_ready_last.py.

Example:
    source_dataset_names:
      - BGL
      - HDFS
    target_dataset_name: TH_1G

Outputs:
    BGL_training_block_w120.npz
    BGL_validation_block_w120.npz
    BGL_testing_block_w120.npz

    HDFS_training_block_w120.npz
    HDFS_validation_block_w120.npz
    HDFS_testing_block_w120.npz

    TH_1G_training_block_w120.npz
    TH_1G_validation_block_w120.npz
    TH_1G_testing_block_w120.npz
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


SEQUENCE_COL = "__logformer_sequence_id__"


# ============================================================
# Configuration
# ============================================================

def load_config(path: str) -> Dict[str, Any]:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("YAML configuration must be a dictionary.")

    return cfg


def get_source_names(cfg: Dict[str, Any]) -> List[str]:
    names = cfg.get("source_dataset_names")

    if not isinstance(names, list):
        raise ValueError("source_dataset_names must be a YAML list.")

    names = [str(x) for x in names]

    if not (1 <= len(names) <= 3):
        raise ValueError(
            "source_dataset_names must contain between 1 and 3 datasets."
        )

    if len(set(names)) != len(names):
        raise ValueError("Duplicate source datasets are not allowed.")

    return names


def get_target_name(cfg: Dict[str, Any]) -> str:
    target = cfg.get("target_dataset_name")

    if target is None:
        raise ValueError("Missing target_dataset_name.")

    return str(target)


def validate_experiment(cfg: Dict[str, Any]) -> Tuple[List[str], str]:
    sources = get_source_names(cfg)
    target = get_target_name(cfg)

    if target in sources:
        raise ValueError(
            f"Target dataset '{target}' cannot also be a source dataset."
        )

    datasets = cfg.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Missing datasets: section in YAML.")

    missing = [
        name
        for name in sources + [target]
        if name not in datasets
    ]

    if missing:
        raise ValueError(
            f"Datasets missing from registry: {missing}. "
            f"Available: {list(datasets.keys())}"
        )

    return sources, target


def get_output_dir(cfg: Dict[str, Any]) -> str:
    return str(
        cfg.get(
            "preprocessed_dir",
            cfg.get("output_dir", "preprocess/preprocessed_data"),
        )
    )


def get_dataset_cfg(
    cfg: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    ds = cfg["datasets"][dataset_name]

    for key in ("train_pkl", "val_pkl", "test_pkl"):
        if ds.get(key) is None:
            raise ValueError(
                f"Dataset '{dataset_name}' missing required key '{key}'."
            )

    return ds


# ============================================================
# Read and normalize PKL
# ============================================================

def read_pkl(
    path: str,
    dataset_name: str,
    split_name: str,
) -> pd.DataFrame:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"{dataset_name} {split_name} PKL not found: {path_obj}"
        )

    print(
        f"Reading {dataset_name} {split_name}: {path_obj}"
    )

    df = pd.read_pickle(path_obj).copy()

    df["__dataset_name__"] = dataset_name
    df["__split_name__"] = split_name

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if (
        "processed_EventTemplate" in df.columns
        and "EventTemplate" not in df.columns
    ):
        df = df.rename(
            columns={
                "processed_EventTemplate": "EventTemplate"
            }
        )

    # Preserve your existing behavior:
    # Original_Label is treated as ground truth when available.
    if "Original_Label" in df.columns:
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])

        df = df.rename(
            columns={"Original_Label": "Label"}
        )

    for col in ("EventTemplate", "Label"):
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
        "-",
        "normal",
        "0",
        "false",
        "benign",
    }


# ============================================================
# Block / sequence handling
# ============================================================

def require_block_col(
    df: pd.DataFrame,
    block_col: str,
    dataset_name: str,
    split_name: str,
):
    if block_col not in df.columns:
        raise ValueError(
            f"{dataset_name} {split_name}: "
            f"'{block_col}' does not exist. "
            f"Available: {list(df.columns)}"
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
    Checks ORIGINAL block IDs before internal prefixes are created.

    This is especially important for HDFS where one BlockId should
    belong to only one split.
    """
    train_ids = set(
        train_df[block_col].astype(str).unique()
    )
    val_ids = set(
        val_df[block_col].astype(str).unique()
    )
    test_ids = set(
        test_df[block_col].astype(str).unique()
    )

    overlaps = {
        "train-validation": train_ids & val_ids,
        "train-test": train_ids & test_ids,
        "validation-test": val_ids & test_ids,
    }

    print(
        f"\n{dataset_name} original {block_col} overlap:"
    )

    has_overlap = False

    for name, ids in overlaps.items():
        print(f"  {name}: {len(ids)}")
        has_overlap = has_overlap or len(ids) > 0

    if strict and has_overlap:
        raise ValueError(
            f"{dataset_name}: overlap detected between "
            "train/validation/test original block IDs."
        )


def assign_existing_blocks(
    df: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    block_col: str,
) -> pd.DataFrame:
    """
    Preserve original Node_block_id and create a safe internal
    sequence ID.

    Example:
        Node_block_id = 100

    internal:
        BGL__train__100
        HDFS__train__100

    Therefore IDs from different datasets can never collide later.
    """
    require_block_col(
        df,
        block_col,
        dataset_name,
        split_name,
    )

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
    Build fixed non-overlapping windows from the current row order.

    Example window_size=120:
        rows 0..119   -> window 0
        rows 120..239 -> window 1
        ...

    No shuffling.
    Final incomplete window is dropped.
    """
    out = df.reset_index(drop=True).copy()

    complete_rows = (
        len(out) // window_size
    ) * window_size

    dropped = len(out) - complete_rows

    if complete_rows == 0:
        raise ValueError(
            f"{dataset_name} {split_name}: "
            f"not enough rows for window_size={window_size}."
        )

    if dropped > 0:
        print(
            f"{dataset_name} {split_name}: "
            f"dropping {dropped} trailing rows "
            "that do not make a complete window."
        )

    out = out.iloc[:complete_rows].copy()

    window_numbers = (
        np.arange(complete_rows) // window_size
    )

    out[SEQUENCE_COL] = [
        f"{dataset_name}__{split_name}__window_{i}"
        for i in window_numbers
    ]

    return out


def prepare_dataset(
    cfg: Dict[str, Any],
    dataset_name: str,
    window_size: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    ds_cfg = get_dataset_cfg(
        cfg,
        dataset_name,
    )

    block_col = str(
        ds_cfg.get(
            "block_col",
            cfg.get("block_col", "Node_block_id"),
        )
    )

    sequence_mode = str(
        ds_cfg.get(
            "sequence_mode",
            "existing_blocks",
        )
    )

    block_length_mode = str(
        ds_cfg.get(
            "block_length_mode",
            "keep_variable",
        )
    )

    strict_overlap = bool(
        ds_cfg.get(
            "strict_split_overlap_check",
            cfg.get(
                "strict_split_overlap_check",
                False,
            ),
        )
    )

    if sequence_mode not in {
        "existing_blocks",
        "fixed_nonoverlap_rows",
    }:
        raise ValueError(
            f"{dataset_name}: sequence_mode must be "
            "'existing_blocks' or 'fixed_nonoverlap_rows'."
        )

    if block_length_mode not in {
        "exact",
        "keep_variable",
        "pad_truncate",
    }:
        raise ValueError(
            f"{dataset_name}: invalid block_length_mode."
        )

    print("\n============================================================")
    print(f"PREPROCESS DATASET: {dataset_name}")
    print("============================================================")
    print(f"sequence_mode: {sequence_mode}")
    print(f"block_length_mode: {block_length_mode}")
    print(f"window_size: {window_size}")
    print(f"block_col: {block_col}")
    print("============================================================")

    train_df = normalize_columns(
        read_pkl(
            ds_cfg["train_pkl"],
            dataset_name,
            "train",
        )
    )

    val_df = normalize_columns(
        read_pkl(
            ds_cfg["val_pkl"],
            dataset_name,
            "validation",
        )
    )

    test_df = normalize_columns(
        read_pkl(
            ds_cfg["test_pkl"],
            dataset_name,
            "testing",
        )
    )

    if sequence_mode == "existing_blocks":
        require_block_col(
            train_df,
            block_col,
            dataset_name,
            "train",
        )
        require_block_col(
            val_df,
            block_col,
            dataset_name,
            "validation",
        )
        require_block_col(
            test_df,
            block_col,
            dataset_name,
            "testing",
        )

        check_split_overlap(
            dataset_name,
            train_df,
            val_df,
            test_df,
            block_col,
            strict_overlap,
        )

        train_df = assign_existing_blocks(
            train_df,
            dataset_name,
            "train",
            block_col,
        )
        val_df = assign_existing_blocks(
            val_df,
            dataset_name,
            "validation",
            block_col,
        )
        test_df = assign_existing_blocks(
            test_df,
            dataset_name,
            "testing",
            block_col,
        )

    else:
        train_df = assign_fixed_nonoverlap_windows(
            train_df,
            dataset_name,
            "train",
            window_size,
        )
        val_df = assign_fixed_nonoverlap_windows(
            val_df,
            dataset_name,
            "validation",
            window_size,
        )
        test_df = assign_fixed_nonoverlap_windows(
            test_df,
            dataset_name,
            "testing",
            window_size,
        )

        # Windows were built to exact size.
        block_length_mode = "exact"

    return (
        train_df,
        val_df,
        test_df,
        block_length_mode,
    )


# ============================================================
# Embeddings
# ============================================================

def add_vectors_for_dataset(
    dfs: List[pd.DataFrame],
    model: SentenceTransformer,
    batch_size: int,
) -> List[pd.DataFrame]:
    """
    Reuse Vector if all three PKLs already contain it.
    Otherwise embed unique EventTemplate values across the
    train/validation/test splits of this dataset.
    """
    if all("Vector" in df.columns for df in dfs):
        print(
            "Vector column exists in all splits. "
            "Reusing existing embeddings."
        )
        return dfs

    all_templates = pd.concat(
        [
            df["EventTemplate"]
            for df in dfs
        ],
        ignore_index=True,
    )

    all_templates = (
        all_templates
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        f"Encoding {len(all_templates)} unique templates"
    )

    vectors = model.encode(
        all_templates.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    lookup = dict(
        zip(
            all_templates,
            vectors,
        )
    )

    outputs = []

    for df in dfs:
        out = df.copy()

        out["Vector"] = (
            out["EventTemplate"]
            .astype(str)
            .map(lookup)
        )

        if out["Vector"].isna().any():
            raise ValueError(
                "Some EventTemplate values could not "
                "be mapped to embeddings."
            )

        outputs.append(out)

    return outputs


# ============================================================
# Save NPZ
# ============================================================

def make_npz(
    df: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    output_dir: str,
    window_size: int,
    block_length_mode: str,
):
    """
    One NPZ entry = one complete sequence/block.

    Label encoding:
        [1, 0] -> normal
        [0, 1] -> anomaly

    A sequence/block is anomalous if at least one row inside
    it is anomalous.
    """
    x_data = []
    y_data = []

    normal_count = 0
    anomaly_count = 0
    skipped = 0
    padded = 0
    truncated = 0

    groups = df.groupby(
        SEQUENCE_COL,
        sort=False,
    )

    for _, block_df in tqdm(
        groups,
        desc=f"{dataset_name} {split_name}",
    ):
        vectors = np.asarray(
            block_df["Vector"].tolist(),
            dtype=np.float32,
        )

        if (
            vectors.ndim != 2
            or len(vectors) == 0
        ):
            skipped += 1
            continue

        if block_length_mode == "exact":
            if len(vectors) != window_size:
                skipped += 1
                continue

            x_item = vectors

        elif block_length_mode == "keep_variable":
            x_item = vectors

        else:
            # pad_truncate
            if len(vectors) >= window_size:
                x_item = vectors[:window_size]

                if len(vectors) > window_size:
                    truncated += 1

            else:
                pad_len = (
                    window_size - len(vectors)
                )

                pad = np.zeros(
                    (
                        pad_len,
                        vectors.shape[1],
                    ),
                    dtype=np.float32,
                )

                x_item = np.vstack(
                    [vectors, pad]
                )

                padded += 1

        labels = block_df["Label"].tolist()

        if all(
            is_normal_label(v)
            for v in labels
        ):
            y_item = [1, 0]
            normal_count += 1

        else:
            y_item = [0, 1]
            anomaly_count += 1

        x_data.append(x_item)
        y_data.append(y_item)

    if block_length_mode == "keep_variable":
        x_data = np.array(
            x_data,
            dtype=object,
        )
    else:
        x_data = np.asarray(
            x_data,
            dtype=np.float32,
        )

    y_data = np.asarray(
        y_data,
        dtype=np.float32,
    )

    if len(y_data) == 0:
        raise ValueError(
            f"{dataset_name} {split_name}: "
            "generated dataset is empty."
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    out_path = (
        Path(output_dir)
        / f"{dataset_name}_{split_name}_block_w{window_size}.npz"
    )

    np.savez(
        out_path,
        x=x_data,
        y=y_data,
    )

    print(f"\nSaved: {out_path}")
    print(f"  total sequences: {len(y_data)}")
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
        default="preprocess/config_cross_dataset_last.yml",
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    source_names, target_name = (
        validate_experiment(cfg)
    )

    window_size = int(
        cfg.get("window_size", 120)
    )

    output_dir = get_output_dir(
        cfg
    )

    embedding_model_name = str(
        cfg.get(
            "embedding_model",
            "distilbert-base-nli-mean-tokens",
        )
    )

    embed_batch_size = int(
        cfg.get("embed_batch_size", 128)
    )

    configured_device = cfg.get("device")

    if configured_device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = str(configured_device)

    dataset_order = (
        source_names
        + [target_name]
    )

    # Avoid duplicate processing just in case.
    dataset_order = list(
        dict.fromkeys(dataset_order)
    )

    print("\n============================================================")
    print("LOGFORMER MULTI-SOURCE PREPROCESSING")
    print("============================================================")
    print("Sources:", source_names)
    print("Target:", target_name)
    print("Window size:", window_size)
    print("Output:", output_dir)
    print("NO source concatenation here.")
    print("NO target fraction here.")
    print("============================================================")

    print(
        f"\nLoading embedding model: "
        f"{embedding_model_name}"
    )

    embedding_model = SentenceTransformer(
        embedding_model_name,
        device=device,
    )

    for dataset_name in dataset_order:
        (
            train_df,
            val_df,
            test_df,
            block_length_mode,
        ) = prepare_dataset(
            cfg,
            dataset_name,
            window_size,
        )

        (
            train_df,
            val_df,
            test_df,
        ) = add_vectors_for_dataset(
            [
                train_df,
                val_df,
                test_df,
            ],
            embedding_model,
            embed_batch_size,
        )

        make_npz(
            train_df,
            dataset_name,
            "training",
            output_dir,
            window_size,
            block_length_mode,
        )

        make_npz(
            val_df,
            dataset_name,
            "validation",
            output_dir,
            window_size,
            block_length_mode,
        )

        make_npz(
            test_df,
            dataset_name,
            "testing",
            output_dir,
            window_size,
            block_length_mode,
        )

        # Free large frames before next dataset.
        del train_df, val_df, test_df

    print("\n============================================================")
    print("PREPROCESSING FINISHED")
    print("============================================================")
    print("Sources:", source_names)
    print("Target:", target_name)
    print(
        "All datasets were saved independently. "
        "Target fraction will be sampled during tuning."
    )
    print("============================================================")


if __name__ == "__main__":
    main()