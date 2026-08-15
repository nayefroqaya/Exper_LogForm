#!/usr/bin/env python3
"""
Dataset-name based PKL selector + preprocessor for LogFormer.

Run:
    python preprocess_dataset_selector_from_config_no_overlap.py --config config_cross_dataset.yml

In-domain:
    Uses one dataset's train/val/test PKLs.

Cross-dataset:
    source train PKLs from one/two/three datasets
    + fraction of normal target train BLOCKS
    validation = target val PKL
    testing = target test PKL

Important:
- Uses Original_Label as Label when Original_Label exists.
- Samples normal target data by Node_block_id blocks, not rows.
- In cross_dataset mode, prefixes Node_block_id with dataset/split before concatenation to prevent collisions.
- Does not delete any rows.
- Supports block_length_mode: exact, keep_variable, pad_truncate.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip install pyyaml") from exc


def load_config(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("config.yml must contain a YAML dictionary.")

    return cfg


def require(cfg: Dict[str, Any], key: str):
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"Missing required config key: {key}")
    return cfg[key]


def get_dataset_paths(cfg: Dict[str, Any], dataset_name: str) -> Dict[str, str]:
    datasets = require(cfg, "datasets")
    if dataset_name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found in config datasets. Available: {available}")

    ds = datasets[dataset_name]
    for key in ["train_pkl", "val_pkl", "test_pkl"]:
        if key not in ds or ds[key] is None:
            raise ValueError(f"Dataset '{dataset_name}' missing {key}")

    return {
        "train_pkl": str(ds["train_pkl"]),
        "val_pkl": str(ds["val_pkl"]),
        "test_pkl": str(ds["test_pkl"]),
    }


def read_pkl(path: str, dataset_name: str, split: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{dataset_name} {split} PKL not found: {path_obj}")

    print(f"Reading {dataset_name} {split}: {path_obj}")
    df = pd.read_pickle(path_obj)
    df["__dataset_name__"] = dataset_name
    df["__split_name__"] = split
    return df


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "processed_EventTemplate" in df.columns and "EventTemplate" not in df.columns:
        df = df.rename(columns={"processed_EventTemplate": "EventTemplate"})

    if "Original_Label" in df.columns:
        print("Original_Label unique:", df["Original_Label"].unique())
        if "Label" in df.columns:
            print("Old Label unique:", df["Label"].unique())
            df = df.drop(columns=["Label"])
        df = df.rename(columns={"Original_Label": "Label"})

    required = ["EventTemplate", "Label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available columns: {list(df.columns)}")

    return df


def ensure_block_col(df: pd.DataFrame, block_col: str) -> pd.DataFrame:
    if block_col not in df.columns:
        raise ValueError(f"{block_col} does not exist. Available columns: {list(df.columns)}")
    return df


def prefix_block_ids(df: pd.DataFrame, block_col: str, dataset_name: str, split_name: str) -> pd.DataFrame:
    """
    Prevent Node_block_id collisions when different datasets are concatenated.

    This does NOT delete or reorder rows.
    It only changes the grouping key from:
        12345
    to:
        BGL__train__12345
    """
    df = df.copy()
    df[block_col] = (
        str(dataset_name)
        + "__"
        + str(split_name)
        + "__"
        + df[block_col].astype(str)
    )
    return df


def is_normal_label(value) -> bool:
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s in {"-", "normal", "0", "false", "benign"}


def check_overlap(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame, block_col: str):
    ids_a = set(df_a[block_col].astype(str).unique())
    ids_b = set(df_b[block_col].astype(str).unique())
    overlap = ids_a.intersection(ids_b)

    if overlap:
        print(f"WARNING: {name_a} and {name_b} share {len(overlap)} {block_col} values.")
    else:
        print(f"No {block_col} overlap between {name_a} and {name_b}.")


def select_normal_blocks(df: pd.DataFrame, block_col: str, fraction: float, seed: int) -> pd.DataFrame:
    if fraction <= 0:
        print("target_normal_fraction is 0. No target train normal blocks will be added.")
        return df.iloc[0:0].copy()

    if fraction > 1:
        raise ValueError("target_normal_fraction must be between 0 and 1.")

    normal_block_ids = []
    for block_id, block_df in df.groupby(block_col, sort=False):
        if block_df["Label"].map(is_normal_label).all():
            normal_block_ids.append(block_id)

    print(f"Target train normal blocks available: {len(normal_block_ids)}")

    if len(normal_block_ids) == 0:
        print("WARNING: No normal target train blocks found.")
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    sample_size = int(round(len(normal_block_ids) * fraction))
    sample_size = max(1, sample_size) if fraction > 0 else 0
    sample_size = min(sample_size, len(normal_block_ids))

    sampled_ids = rng.choice(normal_block_ids, size=sample_size, replace=False)
    sampled = df[df[block_col].isin(sampled_ids)].copy()

    print(f"Target normal fraction: {fraction}")
    print(f"Selected target normal blocks: {sample_size}")
    print(f"Selected target normal rows: {len(sampled)}")

    return sampled


def add_or_reuse_vectors(
    dfs: List[pd.DataFrame],
    model_name: str,
    batch_size: int,
    device: str,
) -> List[pd.DataFrame]:
    if all("Vector" in df.columns for df in dfs):
        print("Vector column found in all dataframes. Reusing existing vectors.")
        return dfs

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    all_templates = pd.concat([df["EventTemplate"] for df in dfs], ignore_index=True)
    all_templates = all_templates.astype(str).dropna().unique()

    print(f"Encoding unique templates: {len(all_templates)}")
    embeddings = model.encode(
        all_templates.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
    )

    template_to_vector = dict(zip(all_templates, embeddings))

    out = []
    for df in dfs:
        df = df.copy()
        df["Vector"] = df["EventTemplate"].astype(str).map(template_to_vector)
        missing = df["Vector"].isna().sum()
        if missing > 0:
            raise ValueError(f"{missing} rows could not be mapped to vectors.")
        out.append(df)

    return out


def make_npz_by_block(
    df: pd.DataFrame,
    mode: str,
    log_name: str,
    output_dir: str,
    block_col: str,
    window_size: int,
    block_length_mode: str = "exact",
):
    """
    Convert blocks to NPZ.

    block_length_mode:
      exact:
        Keep only blocks with exactly window_size rows.
        Use this for already fixed-window datasets.

      keep_variable:
        Save full variable-length block sequences.
        DataGenerator pads/truncates to window_size during training.
        Use this for HDFS BlockId data.

      pad_truncate:
        Pad/truncate every block to window_size during preprocessing.
    """
    x_data, y_data = [], []
    skipped = 0
    padded = 0
    truncated = 0
    kept_variable = 0

    if block_length_mode not in {"exact", "keep_variable", "pad_truncate"}:
        raise ValueError("block_length_mode must be one of: exact, keep_variable, pad_truncate")

    for block_id, block_df in tqdm(df.groupby(block_col, sort=False), desc=f"{mode} blocks"):
        vectors = np.array(block_df["Vector"].tolist(), dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            skipped += 1
            continue

        if block_length_mode == "exact":
            if len(vectors) != window_size:
                skipped += 1
                continue
            x_item = vectors

        elif block_length_mode == "keep_variable":
            x_item = vectors
            kept_variable += 1

        else:
            if len(vectors) >= window_size:
                x_item = vectors[:window_size]
                if len(vectors) > window_size:
                    truncated += 1
            else:
                pad_len = window_size - len(vectors)
                pad = np.zeros((pad_len, vectors.shape[1]), dtype=np.float32)
                x_item = np.vstack([vectors, pad])
                padded += 1

        labels = block_df["Label"].tolist()
        if all(is_normal_label(label) for label in labels):
            y = [1, 0]
        else:
            y = [0, 1]

        x_data.append(x_item)
        y_data.append(y)

    if block_length_mode == "keep_variable":
        x_data = np.array(x_data, dtype=object)
    else:
        x_data = np.array(x_data, dtype=np.float32)

    y_data = np.array(y_data, dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{log_name}_{mode}_block_w{window_size}.npz"
    np.savez(out_path, x=x_data, y=y_data)

    print(f"Saved {mode}: {out_path}")
    print(f"  block_length_mode: {block_length_mode}")
    print(f"  x shape: {x_data.shape}")
    print(f"  y shape: {y_data.shape}")
    print(f"  skipped blocks: {skipped}")
    print(f"  padded blocks: {padded}")
    print(f"  truncated blocks: {truncated}")
    print(f"  kept variable-length blocks: {kept_variable}")

def build_in_domain(cfg: Dict[str, Any]):
    dataset_name = require(cfg, "in_domain_dataset_name")
    block_col = cfg.get("block_col", "Node_block_id")

    paths = get_dataset_paths(cfg, dataset_name)

    df_train = ensure_block_col(normalize_label_column(read_pkl(paths["train_pkl"], dataset_name, "train")), block_col)
    df_val = ensure_block_col(normalize_label_column(read_pkl(paths["val_pkl"], dataset_name, "val")), block_col)
    df_test = ensure_block_col(normalize_label_column(read_pkl(paths["test_pkl"], dataset_name, "test")), block_col)

    check_overlap("train", df_train, "val", df_val, block_col)
    check_overlap("train", df_train, "test", df_test, block_col)
    check_overlap("val", df_val, "test", df_test, block_col)

    return df_train, df_val, df_test


def build_cross_dataset(cfg: Dict[str, Any]):
    source_names = require(cfg, "source_dataset_names")
    target_name = require(cfg, "target_dataset_name")

    if not isinstance(source_names, list) or len(source_names) == 0:
        raise ValueError("source_dataset_names must be a non-empty YAML list.")

    block_col = cfg.get("block_col", "Node_block_id")
    fraction = float(cfg.get("target_normal_fraction", 0.0))
    seed = int(cfg.get("seed", 123))

    source_train_dfs = []
    for src_name in source_names:
        paths = get_dataset_paths(cfg, src_name)
        df_src_train = read_pkl(paths["train_pkl"], src_name, "train")
        df_src_train = normalize_label_column(df_src_train)
        df_src_train = ensure_block_col(df_src_train, block_col)

        # Important fix:
        # Prefix source block IDs so they cannot collide with other source datasets
        # or with sampled target-normal blocks after concatenation.
        # This keeps all rows; it only changes the grouping key.
        df_src_train = prefix_block_ids(df_src_train, block_col, src_name, "train")
        source_train_dfs.append(df_src_train)

    target_paths = get_dataset_paths(cfg, target_name)
    df_target_train = ensure_block_col(
        normalize_label_column(read_pkl(target_paths["train_pkl"], target_name, "train")),
        block_col,
    )
    df_target_val = ensure_block_col(
        normalize_label_column(read_pkl(target_paths["val_pkl"], target_name, "val")),
        block_col,
    )
    df_target_test = ensure_block_col(
        normalize_label_column(read_pkl(target_paths["test_pkl"], target_name, "test")),
        block_col,
    )

    target_normal = select_normal_blocks(df_target_train, block_col, fraction, seed)

    # Important fix:
    # Prefix sampled target-normal block IDs before appending to source training data.
    # This prevents accidental merging with source dataset blocks.
    # This keeps all sampled rows.
    target_normal = prefix_block_ids(
        target_normal,
        block_col,
        target_name,
        "target_train_normal_sample",
    )

    df_train = pd.concat(source_train_dfs + [target_normal], ignore_index=True)
    df_val = df_target_val
    df_test = df_target_test

    print("Cross-dataset summary:")
    print(f"  Sources: {source_names}")
    print(f"  Target: {target_name}")
    print(f"  Training rows after append: {len(df_train)}")
    print(f"  Validation rows: {len(df_val)}")
    print(f"  Testing rows: {len(df_test)}")
    print(f"  Training unique {block_col} after prefixing: {df_train[block_col].nunique()}")

    for i, src_df in enumerate(source_train_dfs):
        check_overlap(f"source:{source_names[i]}", src_df, "target_normal_sample", target_normal, block_col)
    for i in range(len(source_train_dfs)):
        for j in range(i + 1, len(source_train_dfs)):
            check_overlap(f"source:{source_names[i]}", source_train_dfs[i], f"source:{source_names[j]}", source_train_dfs[j], block_col)

    return df_train, df_val, df_test


def print_config(cfg: Dict[str, Any]):
    print("========== Effective Configuration ==========")
    for key, value in cfg.items():
        if key == "datasets":
            print("datasets:")
            for ds_name, paths in value.items():
                print(f"  {ds_name}:")
                print(f"    train_pkl: {paths.get('train_pkl')}")
                print(f"    val_pkl: {paths.get('val_pkl')}")
                print(f"    test_pkl: {paths.get('test_pkl')}")
        else:
            print(f"{key}: {value}")
    print("=============================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.get("device") is None:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    cfg["window_size"] = int(cfg.get("window_size", 120))
    cfg["seed"] = int(cfg.get("seed", 123))
    cfg["embed_batch_size"] = int(cfg.get("embed_batch_size", 128))
    cfg["target_normal_fraction"] = float(cfg.get("target_normal_fraction", 0.0))
    cfg["block_length_mode"] = cfg.get("block_length_mode", "exact")

    print_config(cfg)

    setting = cfg.get("setting", "in_domain")
    if setting == "in_domain":
        df_train, df_val, df_test = build_in_domain(cfg)
    elif setting == "cross_dataset":
        df_train, df_val, df_test = build_cross_dataset(cfg)
    else:
        raise ValueError("setting must be either in_domain or cross_dataset")

    df_train, df_val, df_test = add_or_reuse_vectors(
        [df_train, df_val, df_test],
        model_name=cfg.get("embedding_model", "distilbert-base-nli-mean-tokens"),
        batch_size=cfg["embed_batch_size"],
        device=cfg["device"],
    )

    make_npz_by_block(
        df_train, "training", cfg["log_name"], cfg["output_dir"], cfg["block_col"], cfg["window_size"], cfg["block_length_mode"]
    )
    make_npz_by_block(
        df_val, "validation", cfg["log_name"], cfg["output_dir"], cfg["block_col"], cfg["window_size"], cfg["block_length_mode"]
    )
    make_npz_by_block(
        df_test, "testing", cfg["log_name"], cfg["output_dir"], cfg["block_col"], cfg["window_size"], cfg["block_length_mode"]
    )


if __name__ == "__main__":
    main()
