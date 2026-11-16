"""Build the original LogFormer semantic event vectors from trusted PKL splits.

Matches preprocess_bgl.py / preprocess_hdfs.py: SentenceTransformer
'distilbert-base-nli-mean-tokens' applied to parsed EventTemplate text.
Does NOT learn embeddings from labels, group/recreate sequence IDs, or change
LogFormer Model. Use only trusted PKL files (pickle is unsafe for untrusted data).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

MODEL_NAME = 'distilbert-base-nli-mean-tokens'


def nonblank(x):
    return x is not None and not pd.isna(x) and str(x).strip() != ''


def keytext(x):
    return str(x).strip()


def add(mapping, event, template, source):
    if not nonblank(event) or not nonblank(template):
        raise ValueError(f'{source}: missing EventId or EventTemplate')
    event, template = keytext(event), keytext(template)
    if event in mapping and mapping[event] != template:
        raise ValueError(f'{source}: EventId {event!r} has conflicting template text: '
                         f'{mapping[event]!r} vs {template!r}')
    mapping[event] = template


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_name', required=True)
    p.add_argument('--data_root', default='../../LWADLS/datasets')
    p.add_argument('--event-column', default='EventId')
    p.add_argument('--template-column', default='EventTemplate')
    p.add_argument('--id-column', default='Node_block_id')
    p.add_argument('--templates-csv', default=None,
                   help='Original parsed template CSV with EventId and EventTemplate')
    p.add_argument('--model-name', default=MODEL_NAME)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output', default=None)
    args = p.parse_args()
    if args.batch_size < 1: p.error('--batch-size must be positive')

    root = Path(args.data_root) / args.dataset_name / f'1_{args.dataset_name}_Splitted_Datasets'
    mapping, split_ids = {}, {}
    all_events = set()
    # Fail closed on duplicates across splits and on null IDs.
    for split, fname in [('train','train_df.pkl'),('val','val_df.pkl'),('test','test_df.pkl')]:
        path = root / fname
        if not path.is_file(): raise FileNotFoundError(path)
        df = pd.read_pickle(path)
        if args.id_column not in df.columns or args.event_column not in df.columns:
            raise ValueError(f'{path}: required {args.id_column}, {args.event_column}; got {list(df.columns)}')
        if df[args.id_column].isna().any() or df[args.event_column].isna().any():
            raise ValueError(f'{path}: null sequence or event IDs')
        ids = set(df[args.id_column].astype(str))
        if any(not x.strip() for x in ids): raise ValueError(f'{path}: blank Node_block_id')
        for other, other_ids in split_ids.items():
            overlap = ids & other_ids
            if overlap: raise ValueError(f'Node_block_id cross-split overlap {split}/{other}: {list(overlap)[:5]}')
        split_ids[split] = ids
        if args.template_column in df.columns:
            for eid, tmpl in tqdm(zip(df[args.event_column], df[args.template_column]),
                                  total=len(df), desc=f'{split}: map event templates'):
                add(mapping, eid, tmpl, str(path))
        all_events.update(df[args.event_column].astype(str))
        print(f'{split}: {len(df)} event rows, {len(ids)} original block IDs', flush=True)

    if args.templates_csv:
        csv_path = Path(args.templates_csv)
        table = pd.read_csv(csv_path)
        for col in (args.event_column, args.template_column):
            if col not in table.columns:
                raise ValueError(f'{csv_path}: missing {col}; found {list(table.columns)}')
        for eid, tmpl in zip(table[args.event_column],table[args.template_column]):
            add(mapping,eid,tmpl,str(csv_path))

    missing = sorted(all_events - mapping.keys())
    if missing:
        raise ValueError(
            f'{len(missing)} EventId values lack EventTemplate text; examples: {missing[:8]}. '
            f'PKL files must include {args.template_column}, OR supply '
            '--templates-csv PATH/TO/ORIGINAL.log_templates.csv. '
            'EventId strings are NOT valid substitutes for template text.'
        )

    # Encode only the templates used in the datasets. The pretrained encoder
    # is frozen; no label access, corpus fitting, or test-based model selection.
    events = sorted(all_events)
    texts = [mapping[eid] for eid in events]
    chosen = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else (
        'cpu' if args.device == 'auto' else args.device)
    if chosen == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    print(f'Encoding {len(events)} unique EventId templates with {args.model_name} on {chosen}', flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_name, device=chosen)
    vectors = model.encode(texts, batch_size=args.batch_size,
                           show_progress_bar=True, convert_to_numpy=True,
                           normalize_embeddings=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(events), 768) or not np.isfinite(vectors).all():
        raise ValueError(f'Original LogFormer encoder must produce (N,768); got {vectors.shape}')

    dest = Path(args.output or f'logformer_embeddings/{args.dataset_name}_original_encoder.npz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest, event_ids=np.asarray(events,dtype=str), embeddings=vectors)
    meta = {
        'dataset': args.dataset_name, 'encoder': args.model_name, 'dimension': 768,
        'unique_event_ids':len(events),'split_sequence_ids': {s: len(v) for s,v in split_ids.items()},
        'template_source': 'PKL EventTemplate and/or supplied original template CSV',
        'embedding_method':'original LogFormer SentenceTransformer.encode(EventTemplate)',
        'labels_used':False, 'parameters_fitted_on_splits':False,
        'note':'Encoding pre-existing templates from held-out splits with the frozen pretrained encoder does not use labels, but records their template text; document this preprocessing policy.'
    }
    dest.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
    print('Saved:',dest, flush=True)

if __name__=='__main__': main()
