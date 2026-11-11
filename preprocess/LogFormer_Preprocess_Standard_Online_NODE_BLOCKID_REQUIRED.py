#!/usr/bin/env python3
"""
Unified LogFormer preprocessing for ORIGINAL Standard + bounded-window Online.
All datasets: Node_block_id REQUIRED across train/validation/test; fail fast on any cross-split overlap.

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
- Each NPZ also stores first_anomaly_step (1-based; -1 for normal/no onset in retained sequence).
- Target fraction sampling happens only in the tuning stage when cross-dataset mode is used.
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
REQUIRED_ORIGINAL_BLOCK_ID = "Node_block_id"


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

    # Original_Label is treated as ground truth when present; never a model input.
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


# Names are considered only when already present in the ORIGINAL input PKLs.
# IDs generated by LogFormer, especially split-prefixed IDs, are never audited.
ORIGINAL_ID_CANDIDATES = (
    "Node_block_id", "BlockId", "block_id", "blockid", "Block_ID",
    "SessionId", "session_id", "sessionid", "Session_ID",
    "SequenceId", "sequence_id", "sequenceid",
    "TraceId", "trace_id", "traceid",
)


def resolve_original_id_column(
    dataset_name: str,
    splits: Dict[str, pd.DataFrame],
    requested_column: str | None = None,
    required: bool = False,
) -> str | None:
    """Choose an ORIGINAL identifier present consistently in all three raw splits.

    Explicit identifier selection always fails closed if the column is missing.
    For fixed, row-windowed datasets with no source identifier, audit status is
    SKIPPED_NO_ORIGINAL_ID, never PASS/verified leakage-free.
    """
    requested = str(requested_column).strip() if requested_column is not None else "auto"
    all_colnames = {name: list(df.columns) for name, df in splits.items()}

    if requested.lower() != "auto":
        missing = [name for name, cols in all_colnames.items() if requested not in cols]
        if missing:
            raise ValueError(
                f"{dataset_name}: explicitly configured overlap_id_col={requested!r} "
                f"is missing from {missing}. Stop: cannot verify original IDs. "
                f"Columns: {all_colnames}"
            )
        return requested

    # Automatic candidate detection is conservative and case-insensitive.
    found = []
    for candidate in ORIGINAL_ID_CANDIDATES:
        locations = {}
        for split_name, cols in all_colnames.items():
            matches = [c for c in cols if c.casefold() == candidate.casefold()]
            if matches:
                locations[split_name] = matches[0]
        if locations:
            found.append((candidate, locations))

    for candidate, locations in found:
        if len(locations) == len(splits):
            spellings = set(locations.values())
            if len(spellings) != 1:
                raise ValueError(
                    f"{dataset_name}: original ID column has inconsistent "
                    f"capitalization/spelling across splits: {locations}. "
                    "Specify a consistent overlap_id_col explicitly."
                )
            return next(iter(spellings))

    if found:
        raise ValueError(
            f"{dataset_name}: a possible original BlockId/SessionId exists "
            f"only in some splits: {found}. Refusing to skip the audit."
        )

    if required:
        raise ValueError(
            f"{dataset_name}: no original BlockId/SessionId column found "
            "in all three splits. Specify overlap_id_col or provide source IDs."
        )
    return None


def check_original_ids_strict(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report_path: Path,
    requested_column: str | None = None,
    require_id: bool = False,
) -> Dict[str, Any]:
    """Always stop on overlapping original IDs, for ANY selected dataset.

    Same original ID repeated within one split is expected and permitted.
    Identical log messages across independent IDs are not considered ID overlap.
    Reports are written for PASS, overlap, invalid and skipped cases.
    """
    import json

    splits = {"train": train_df, "validation": val_df, "test": test_df}
    report_path = Path(report_path)
    report = {
        "dataset": dataset_name,
        "result": "NOT_CHECKED",
        "id_column": None,
        "counts": {},
        "sample_overlap_ids": {},
        "note": "Compares original input-PKL IDs before any split-prefix or block grouping.",
    }

    def save_report():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        id_column = resolve_original_id_column(
            dataset_name, splits, requested_column, required=require_id
        )
    except ValueError as exc:
        report["result"] = "FAILED_ID_SCHEMA"
        report["error"] = str(exc)
        save_report()
        raise

    report["id_column"] = id_column
    print("\n" + "=" * 76)
    print(f"ORIGINAL-ID SPLIT AUDIT | {dataset_name}")
    print("=" * 76)

    if id_column is None:
        report["result"] = "SKIPPED_NO_ORIGINAL_ID"
        report["note"] = (
            "No BlockId/SessionId/SequenceId is present in the original input "
            "PKLs. ID overlap cannot be verified for fixed-row windowing."
        )
        save_report()
        print("NOT APPLICABLE: no source ID column in the input PKLs.")
        print("ID overlap was NOT verified; only split-specific row windows are used.")
        print(f"Audit report: {report_path}")
        print("=" * 76)
        return report

    print(f"Original ID column: {id_column}")
    id_sets = {}
    for name, df in splits.items():
        values = pd.Series(pd.unique(df[id_column]), dtype="object")
        normalized = values.astype("string").str.strip()
        invalid = normalized.isna() | normalized.eq("") | normalized.str.casefold().isin(
            ("nan", "none", "null", "<na>")
        )
        invalid = invalid.fillna(True)
        if bool(invalid.any()):
            report["result"] = "FAILED_INVALID_ID"
            report["error"] = (
                f"{dataset_name}: {name} has missing/blank original {id_column} IDs"
            )
            report["counts"][f"{name}_invalid_unique_ids"] = int(invalid.sum())
            save_report()
            raise ValueError(report["error"] + f". Audit: {report_path}")
        id_sets[name] = set(normalized.tolist())
        report["counts"][f"{name}_unique_ids"] = len(id_sets[name])

    failures = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        key = f"{left}_vs_{right}"
        overlap = id_sets[left] & id_sets[right]
        report["counts"][f"{key}_overlaps"] = len(overlap)
        report["sample_overlap_ids"][key] = sorted(overlap)[:10]
        print(f"{key}: {len(overlap)} shared original IDs")
        if overlap:
            failures.append(f"{key}: {len(overlap)}")

    report["result"] = "FAILED_OVERLAP" if failures else "PASS"
    save_report()
    print(f"Audit report: {report_path}")
    if failures:
        print("*** ALERT: ORIGINAL BLOCK/SESSION ID OVERLAP — STOPPING ***")
        raise ValueError(
            f"{dataset_name}: original {id_column} overlap across splits: "
            + "; ".join(failures)
        )
    print("PASS: original IDs are disjoint across train, validation and test.")
    print("=" * 76)
    return report


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
    print(f"block_col: {block_col}" if sequence_mode == "existing_blocks" else "block_col: N/A (fixed row windows)")
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

    # Universally REQUIRED original BlockId check.  Every source split MUST
    # contain Node_block_id, regardless of dataset and windowing mode.
    # Never skip this audit, and never audit split-prefixed generated IDs.
    audit_report = check_original_ids_strict(
        dataset_name=dataset_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        requested_column=REQUIRED_ORIGINAL_BLOCK_ID,
        require_id=True,
        report_path=Path(get_output_dir(cfg)) / f"{dataset_name}_original_id_overlap_audit.json",
    )
    if audit_report["result"] != "PASS" or audit_report["id_column"] != REQUIRED_ORIGINAL_BLOCK_ID:
        raise RuntimeError(f"{dataset_name}: mandatory Node_block_id audit did not PASS")

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
    """Create LogFormer x/y and *evaluation-only* event-truth metadata.

    IMPORTANT: For pad_truncate, the sequence label is computed from the
    actually observed (retained) events, never from events beyond the
    truncated window. The model never receives labels at inference time.
    """
    x_data, y_data, event_labels_data = [], [], []
    first_anomaly_steps, sequence_lengths, sequence_ids = [], [], []
    normal_count = anomaly_count = skipped = padded = truncated = oversized = 0

    for group_id, block_df in tqdm(df.groupby(SEQUENCE_COL, sort=False),
                                    desc=f"{dataset_name} {split_name}"):
        vectors = np.asarray(block_df["Vector"].tolist(), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0:
            skipped += 1
            continue

        raw_labels = np.asarray([int(not is_normal_label(z))
                                 for z in block_df["Label"].tolist()], dtype=np.int8)
        if len(raw_labels) != len(vectors):
            raise ValueError(f"{dataset_name}/{split_name}/{group_id}: labels/vector mismatch")

        if block_length_mode == "exact":
            if len(vectors) != window_size:
                skipped += 1
                continue
            x_item = vectors
            length = len(vectors)
        elif block_length_mode == "keep_variable":
            x_item = vectors
            length = len(vectors)
            if length > window_size:
                oversized += 1
        else:
            # pad_truncate. Never derive labels from unseen truncated events.
            length = min(len(vectors), window_size)
            x_item = vectors[:length]
            if len(vectors) > window_size:
                truncated += 1
            elif len(vectors) < window_size:
                pad = np.zeros((window_size - length, vectors.shape[1]), dtype=np.float32)
                x_item = np.vstack([x_item, pad])
                padded += 1

        observed_labels = raw_labels[:length]
        anomaly_positions = np.flatnonzero(observed_labels == 1)
        is_anomaly = bool(len(anomaly_positions))
        onset = int(anomaly_positions[0] + 1) if is_anomaly else -1
        if is_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1

        event_labels = np.full(len(x_item), -1, dtype=np.int8)
        event_labels[:length] = observed_labels

        x_data.append(x_item)
        y_data.append([0., 1.] if is_anomaly else [1., 0.])
        event_labels_data.append(event_labels)
        first_anomaly_steps.append(onset)
        sequence_lengths.append(int(length))
        sequence_ids.append(str(group_id))

    if not y_data:
        raise ValueError(f"{dataset_name} {split_name}: generated NPZ is empty")

    if block_length_mode == "keep_variable":
        # Exactly preserve the original variable-length data format.
        x_output = np.empty(len(x_data), dtype=object)
        evt_output = np.empty(len(event_labels_data), dtype=object)
        for i in range(len(x_data)):
            x_output[i] = x_data[i]
            evt_output[i] = event_labels_data[i]
    else:
        x_output = np.asarray(x_data, dtype=np.float32)
        evt_output = np.asarray(event_labels_data, dtype=np.int8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{dataset_name}_{split_name}_block_w{window_size}.npz"
    np.savez(out_path,
             x=x_output,
             y=np.asarray(y_data, dtype=np.float32),
             first_anomaly_step=np.asarray(first_anomaly_steps, dtype=np.int32),
             event_labels=evt_output,
             sequence_length=np.asarray(sequence_lengths, dtype=np.int32),
             sequence_id=np.asarray(sequence_ids, dtype=str),
             original_id_audit_status=np.asarray("PASS"),
             original_id_audit_column=np.asarray(REQUIRED_ORIGINAL_BLOCK_ID))

    print(f"\nSaved: {out_path}")
    print(f"  total sequences={len(y_data)} | normal={normal_count} | anomaly={anomaly_count}")
    print(f"  skipped={skipped} | padded={padded} | truncated={truncated}")
    if oversized:
        print(f"  WARNING: {oversized} variable-length blocks exceed W={window_size}. ")
        print("  Original Standard model sees only first W while Online sees the full stream.")
        print("  For identical evaluation horizons, change preprocessing segmentation.")
    print("  Saved event_labels, sequence_length, sequence_id, first_anomaly_step.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config_LogFormer_ALL_DATASETS_STRICT.yml",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Optional single dataset name from YAML registry, e.g. BGL, HDFS, TH_1G. "
             "Overrides setting/in_domain_dataset_name for this run.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dataset is not None:
        # --dataset overrides even a cross-dataset YAML setting.
        setting = "in_domain"
        dataset_order = [args.dataset]
        source_or_domain_names = dataset_order
        target_name = None
    else:
        setting, source_or_domain_names, target_name = get_selected_datasets(cfg)
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

    embedding_model = None  # Created only after the first dataset passes the ID audit.

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

        if embedding_model is None:
            print(f"\nLoading embedding model after Node_block_id audit: {embedding_model_name}")
            embedding_model = SentenceTransformer(embedding_model_name, device=device)

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