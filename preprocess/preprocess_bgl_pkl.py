import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def normal_label(value) -> bool:
    """BGL uses '-' for normal logs. Everything else is treated as anomaly."""
    return str(value) == "-"


def build_windows_from_blocks(df: pd.DataFrame, mode: str, log_name: str, output_dir: Path,
                              block_col: str = "Node_block_id", window_size: int = 120) -> None:
    required = {block_col, "Label", "Vector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{mode} dataframe is missing columns: {sorted(missing)}")

    x_data, y_data = [], []
    skipped = 0

    for _, df_blk in tqdm(df.groupby(block_col, sort=False), desc=f"{mode} blocks"):
        # Keep the existing LogFormer setup: one sample = one complete block/window.
        # Blocks not exactly window_size are skipped to keep x shape fixed for the Transformer.
        if len(df_blk) != window_size:
            skipped += 1
            continue

        x_data.append(np.asarray(df_blk["Vector"].tolist(), dtype=np.float32))

        labels = df_blk["Label"].tolist()
        y_data.append([1, 0] if all(normal_label(label) for label in labels) else [0, 1])

    x_data = np.asarray(x_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.float32)

    output_path = output_dir / f"{log_name}_{mode}_block_w{window_size}.npz"
    np.savez(output_path, x=x_data, y=y_data)
    print(f"{mode} saved to {output_path}: x={x_data.shape}, y={y_data.shape}, skipped_blocks={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LogFormer BGL NPZ files directly from split PKL dataframes.")
    parser.add_argument("--pkl_dir", type=str,
                        default="/storage/home/roqaya/Exper_LogForm/dataset/BGL/1_BGL_Splitted_Datasets",
                        help="Directory containing train_df.pkl, val_df.pkl, and test_df.pkl")
    parser.add_argument("--output_dir", type=str, default="preprocess/preprocessed_data",
                        help="Where to write *_block_w*.npz files")
    parser.add_argument("--log_name", type=str, default="BGL")
    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--block_col", type=str, default="Node_block_id")
    parser.add_argument("--text_col", type=str, default="EventTemplate",
                        help="Column used for sentence-transformer embeddings")
    parser.add_argument("--embedding_model", type=str, default="distilbert-base-nli-mean-tokens")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    pkl_dir = Path(os.path.expanduser(args.pkl_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "training": pkl_dir / "train_df.pkl",
        "validation": pkl_dir / "val_df.pkl",
        "testing": pkl_dir / "test_df.pkl",
    }

    for mode, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {mode} PKL file: {path}")

    print("Loading PKL split dataframes directly; Drain/parse_log.py is not used.")
    dataframes = {mode: pd.read_pickle(path) for mode, path in split_paths.items()}

    for mode, df in dataframes.items():
        if args.text_col not in df.columns:
            raise ValueError(f"{mode} dataframe is missing text column: {args.text_col}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.embedding_model, device=device)

    print("Embedding unique EventTemplate values across train/val/test...")
    all_templates = pd.concat(
        [df[args.text_col].astype(str) for df in dataframes.values()],
        ignore_index=True,
    ).dropna().unique()

    embeddings = model.encode(
        all_templates.tolist() if hasattr(all_templates, "tolist") else list(all_templates),
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    template_dict = dict(zip(all_templates, embeddings))

    for mode, df in dataframes.items():
        df = df.copy()
        df["Vector"] = df[args.text_col].astype(str).map(template_dict)
        if df["Vector"].isna().any():
            missing_count = int(df["Vector"].isna().sum())
            raise ValueError(f"{mode} has {missing_count} rows without embeddings")

        build_windows_from_blocks(
            df=df,
            mode=mode,
            log_name=args.log_name,
            output_dir=output_dir,
            block_col=args.block_col,
            window_size=args.window_size,
        )


if __name__ == "__main__":
    main()
