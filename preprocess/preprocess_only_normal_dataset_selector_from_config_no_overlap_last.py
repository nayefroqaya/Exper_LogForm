#!/usr/bin/env python3
"""
Unified LogFormer preprocessing.

Supports BOTH:

1) IN-DOMAIN
   setting: in_domain
   in_domain_dataset_name: BGL

   -> preprocess only BGL train / validation / test

2) CROSS-DATASET
   setting: cross_dataset
   source_dataset_names:
     - BGL
     - HDFS
   target_dataset_name: TH_1G

   -> preprocess every selected source and the target independently

IMPORTANT:
- Datasets are NEVER concatenated during preprocessing.
- No target fraction is sampled during preprocessing.
- Complete train / validation / test NPZ files are written per dataset.
- Target fraction sampling happens only in Frackition_normalandanomal_tune_transformer_pkl_ready_last.py.
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
# Configuration helpers
# ============================================================

def load_config(path: str) -> Dict[str, Any]:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("YAML configuration must contain a dictionary.")

    return cfg


def get_setting(cfg: Dict[str, Any]) -> str:
    setting = str(cfg.get("setting", "in_domain")).strip().lower()

    if setting not in {"in_domain", "cross_dataset"}:
        raise ValueError(
            "setting must be either 'in_domain' or 'cross_dataset'."
        )

    return setting


def get_in_domain_name(cfg: Dict[str, Any]) -> str:
    name = cfg.get("in_domain_dataset_name")

    if name is None:
        raise ValueError(
            "in_domain_dataset_name is required when setting: in_domain."
        )

    return str(name)


def get_source_names(cfg: Dict[str, Any]) -> List[str]:
    names = cfg.get("source_dataset_names")

    if not isinstance(names, list):
        raise ValueError(
            "source_dataset_names must be a YAML list in cross_dataset mode."
        )

    names = [str(x) for x in names]

    if not (1 <= len(names) <= 3):
        raise ValueError(
            "cross_dataset mode supports 1, 2, or 3 source datasets."
        )

    if len(set(names)) != len(names):
        raise ValueError("Duplicate source datasets are not allowed.")

    return names


def get_target_name(cfg: Dict[str, Any]) -> str:
    name = cfg.get("target_dataset_name")

    if name is None:
        raise ValueError(
            "target_dataset_name is required when setting: cross_dataset."
        )

    return str(name)


def get_selected_datasets(
    cfg: Dict[str, Any],
) -> Tuple[str, List[str], str | None]:
    setting = get_setting(cfg)

    if setting == "in_domain":
        dataset_name = get_in_domain_name(cfg)
        return setting, [dataset_name], None

    source_names = get_source_names(cfg)
    target_name = get_target_name(cfg)

    if target_name in source_names:
        raise ValueError(
            f"Target dataset '{target_name}' cannot also be a source dataset."
        )

    return setting, source_names, target_name


def get_output_dir(cfg: Dict[str, Any]) -> str:
    return str(
        cfg.get(
            "preprocessed_dir",
            cfg.get("output_dir", "preprocess/preprocessed_data"),
        )
    )


def validate_dataset_registry(
    cfg: Dict[str, Any],
    dataset_names: List[str],
):
    datasets = cfg.get("datasets")

    if not isinstance(datasets, dict):
        raise ValueError("Missing 'datasets:' registry in YAML.")

    missing = [name for name in dataset_names if name not in datasets]

    if missing:
        raise ValueError(
            f"Datasets missing from registry: {missing}. "
            f"Available: {list(datasets.keys())}"
        )

    for name in dataset_names:
        ds = datasets[name]

        for key in ("train_pkl", "val_pkl", "test_pkl"):
            if ds.get(key) is None:
                raise ValueError(
                    f"Dataset '{name}' is missing required key '{key}'."
                )


def get_dataset_cfg(
    cfg: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    return cfg["datasets"][dataset_name]


# ============================================================
# Read / normalize PKL
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

    print(f"Reading {dataset_name} {split_name}: {path_obj}")

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
            columns={"processed_EventTemplate": "EventTemplate"}
        )

    # Preserve your existing logic:
    # Original_Label is treated as the ground-truth label if present.
    if "Original_Label" in df.columns:
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])

        df = df.rename(columns={"Original_Label": "Label"})

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
    Check ORIGINAL block IDs before internal dataset/split prefixes.

    Useful especially for HDFS:
    the same BlockId should not occur in more than one split.
    """
    train_ids = set(train_df[block_col].astype(str).unique())
    val_ids = set(val_df[block_col].astype(str).unique())
    test_ids = set(test_df[block_col].astype(str).unique())

    overlaps = {
        "train-validation": train_ids & val_ids,
        "train-test": train_ids & test_ids,
        "validation-test": val_ids & test_ids,
    }

    print(f"\n{dataset_name} original {block_col} overlap check:")

    any_overlap = False

    for pair, ids in overlaps.items():
        print(f"  {pair}: {len(ids)}")
        any_overlap = any_overlap or bool(ids)

    if strict and any_overlap:
        raise ValueError(
            f"{dataset_name}: original block IDs overlap across splits."
        )


def assign_existing_blocks(
    df: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    block_col: str,
) -> pd.DataFrame:
    """
    Preserve the original Node_block_id and create a safe internal ID.

    Example:
        BGL train block 100  -> BGL__train__100
        HDFS train block 100 -> HDFS__train__100

    The original Node_block_id column itself is not changed.
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
    Build consecutive non-overlapping windows using current row order.

    Example window_size=120:
        rows   0..119 -> window 0
        rows 120..239 -> window 1
        ...

    The final incomplete window is dropped.
    """
    out = df.reset_index(drop=True).copy()

    complete_rows = (len(out) // window_size) * window_size
    dropped = len(out) - complete_rows

    if complete_rows == 0:
        raise ValueError(
            f"{dataset_name} {split_name}: not enough rows "
            f"for window_size={window_size}."
        )

    if dropped > 0:
        print(
            f"{dataset_name} {split_name}: dropping {dropped} trailing rows "
            "that do not form a complete window."
        )

    out = out.iloc[:complete_rows].copy()

    window_numbers = np.arange(complete_rows) // window_size

    out[SEQUENCE_COL] = [
        f"{dataset_name}__{split_name}__window_{i}"
        for i in window_numbers
    ]

    return out


def prepare_dataset(
    cfg: Dict[str, Any],
    dataset_name: str,
    window_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ds_cfg = get_dataset_cfg(cfg, dataset_name)

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
            cfg.get("strict_split_overlap_check", False),
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
            f"{dataset_name}: block_length_mode must be "
            "exact, keep_variable, or pad_truncate."
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
        read_pkl(ds_cfg["train_pkl"], dataset_name, "train")
    )
    val_df = normalize_columns(
        read_pkl(ds_cfg["val_pkl"], dataset_name, "validation")
    )
    test_df = normalize_columns(
        read_pkl(ds_cfg["test_pkl"], dataset_name, "testing")
    )

    if sequence_mode == "existing_blocks":
        require_block_col(
            train_df, block_col, dataset_name, "train"
        )
        require_block_col(
            val_df, block_col, dataset_name, "validation"
        )
        require_block_col(
            test_df, block_col, dataset_name, "testing"
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

        # Windows are exact by construction.
        block_length_mode = "exact"

    return train_df, val_df, test_df, block_length_mode


# ============================================================
# Embeddings
# ============================================================

def add_vectors_for_dataset(
    dfs: List[pd.DataFrame],
    model: SentenceTransformer,
    batch_size: int,
) -> List[pd.DataFrame]:
    """
    Reuse Vector only if all three splits already contain it.
    Otherwise embed unique EventTemplate values across this dataset.
    """
    if all("Vector" in df.columns for df in dfs):
        print(
            "Vector column found in train/validation/test. "
            "Reusing existing embeddings."
        )
        return dfs

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

        out["Vector"] = (
            out["EventTemplate"]
            .astype(str)
            .map(lookup)
        )

        if out["Vector"].isna().any():
            raise ValueError(
                "Some EventTemplate values could not be embedded."
            )

        outputs.append(out)

    return outputs


# ============================================================
# NPZ conversion
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
    One NPZ entry = one complete block/sequence.

    Label encoding:
        [1, 0] = normal
        [0, 1] = anomaly

    A block/sequence is anomalous if at least one row is anomalous.
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
        desc=f"{dataset_name} {split_name}",
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

        else:
            # pad_truncate
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
            y_item = [1, 0]
            normal_count += 1
        else:
            y_item = [0, 1]
            anomaly_count += 1

        x_data.append(x_item)
        y_data.append(y_item)

    if block_length_mode == "keep_variable":
        x_data = np.array(x_data, dtype=object)
    else:
        x_data = np.asarray(x_data, dtype=np.float32)

    y_data = np.asarray(y_data, dtype=np.float32)

    if len(y_data) == 0:
        raise ValueError(
            f"{dataset_name} {split_name}: generated NPZ is empty."
        )

    os.makedirs(output_dir, exist_ok=True)

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

    cfg = load_config(args.config)

    setting, source_or_domain_names, target_name = (
        get_selected_datasets(cfg)
    )

    if setting == "in_domain":
        dataset_order = source_or_domain_names
    else:
        dataset_order = source_or_domain_names + [target_name]

    # Remove any accidental duplicate while preserving order.
    dataset_order = list(dict.fromkeys(dataset_order))

    validate_dataset_registry(
        cfg,
        dataset_order,
    )

    window_size = int(cfg.get("window_size", 120))
    output_dir = get_output_dir(cfg)

    embedding_model_name = str(
        cfg.get(
            "embedding_model",
            "distilbert-base-nli-mean-tokens",
        )
    )

    embed_batch_size = int(cfg.get("embed_batch_size", 128))

    configured_device = cfg.get("device")

    if configured_device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = str(configured_device)

    print("\n============================================================")
    print("UNIFIED LOGFORMER PREPROCESSING")
    print("============================================================")
    print("Setting:", setting)

    if setting == "in_domain":
        print("In-domain dataset:", dataset_order[0])
    else:
        print("Sources:", source_or_domain_names)
        print("Target:", target_name)

    print("Window size:", window_size)
    print("Output directory:", output_dir)
    print("NO dataset concatenation during preprocessing.")
    print("NO target fraction during preprocessing.")
    print("============================================================")

    print(f"\nLoading embedding model: {embedding_model_name}")

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
            [train_df, val_df, test_df],
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

        del train_df, val_df, test_df

    print("\n============================================================")
    print("PREPROCESSING FINISHED")
    print("============================================================")

    if setting == "in_domain":
        print(
            f"Created train/validation/test NPZ files for "
            f"{dataset_order[0]}."
        )
    else:
        print(
            "Created independent train/validation/test NPZ files "
            "for every source and target dataset."
        )

    print("============================================================")


if __name__ == "__main__":
    main()