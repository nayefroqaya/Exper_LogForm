"""PKL-based, leakage-audited LogFormer training + full-sequence/online test.

IMPORTANT: original 768-dimensional event embeddings are REQUIRED. This script
never invents/retrains them or fits an encoder on validation/test data.
Only open trusted .pkl/.pt inputs (pickle is executable).
"""
import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score
from model import Model


@dataclass
class Sequence:
    sid: str
    events: np.ndarray  # shape [events, 768]
    label: int


def parse_label(value):
    # Matches the previously supplied PKL PLELog preprocessing rule.
    if pd.isna(value):
        raise ValueError('Missing label')
    s = str(value).strip()
    if s.lower() in ('-', 'normal', '0', 'false', 'benign'):
        return 0
    if s.lower() in ('anomaly', 'anomalous', '1', 'true'):
        return 1
    # Original_Label may contain named failure categories such as R_HDA_NR.
    # This mirrors PLELog's explicit original-label mapping; audit for each dataset.
    return 1


def load_embedding_map(path):
    if not path:
        return None
    if Path(path).suffix.lower() != '.npz':
        raise ValueError('Embedding map must be a trusted NPZ with event_ids and embeddings arrays')
    z = np.load(path, allow_pickle=False)
    if not {'event_ids', 'embeddings'}.issubset(z.files):
        raise ValueError('NPZ must contain event_ids and embeddings')
    ids, values = z['event_ids'], z['embeddings']
    if values.ndim != 2 or values.shape[1] != 768 or len(ids) != len(values):
        raise ValueError('Embedding map must have shape (number_of_event_ids, 768)')
    mapping = {}
    for k, v in zip(ids, values):
        key = str(k)
        if key in mapping:
            raise ValueError(f'Duplicate EventId in embedding mapping: {key}')
        mapping[key] = v.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite values in embedding map')
    return mapping


def coerce_embedding(value, sid):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        v = np.asarray(value, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f'Bad event embedding in Node_block_id={sid}: {exc}') from exc
    if v.shape != (768,) or not np.isfinite(v).all():
        raise ValueError(f'Embedding for {sid} must have 768 finite numbers, got {v.shape}')
    return v


def read_sequences(path, args, embedding_map):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    df = pd.read_pickle(path)  # trusted dataset only
    required = [args.id_column, args.event_column, args.label_column]
    if not embedding_map:
        required.append(args.embedding_column)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f'{path}: missing columns {missing}. LogFormer needs original 768-D '
            'embeddings: use --embedding-column or --embedding-map pointing to '
            'the ORIGINAL encoder outputs (do not substitute random vectors). '
            f'Available columns: {list(df.columns)}'
        )
    if df[args.id_column].isna().any():
        raise ValueError(f'{path}: null Node_block_id')
    grouped = {}
    for sid_raw, event_raw, label_raw, *extra in df[required].itertuples(index=False, name=None):
        sid = str(sid_raw)
        if not sid.strip():
            raise ValueError(f'{path}: blank Node_block_id')
        label = parse_label(label_raw)
        events = list(event_raw) if isinstance(event_raw, (list, tuple, np.ndarray)) else [event_raw]
        if not events or any(v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in events):
            raise ValueError(f'{path}: null/empty EventId for {sid}')
        if embedding_map is not None:
            vectors = []
            for ev in events:
                event_id = str(ev)
                if event_id not in embedding_map:
                    raise ValueError(f'EventId {event_id!r} lacks original pretrained embedding; aborting')
                vectors.append(embedding_map[event_id])
        else:
            raw = extra[0]
            if len(events) == 1:
                vectors = [coerce_embedding(raw, sid)]
            else:
                # Multi-event rows must supply one 768-D vector per event.
                if isinstance(raw, str):
                    raw = json.loads(raw)
                arr = np.asarray(raw, dtype=np.float32)
                if arr.shape != (len(events), 768) or not np.isfinite(arr).all():
                    raise ValueError(f'{path}: {sid} expected ({len(events)},768) embedding vectors')
                vectors = list(arr)
        if sid not in grouped:
            grouped[sid] = [[], 0]
        grouped[sid][0].extend(vectors)
        grouped[sid][1] = max(grouped[sid][1], label)
    return [Sequence(k, np.stack(v[0]), v[1]) for k, v in grouped.items()]


def load_splits(args, include_test=True):
    mapping = load_embedding_map(args.embedding_map)
    result, existing = {}, {}
    selected = [('train', 'train_df.pkl'), ('val', 'val_df.pkl')]
    if include_test:
        selected.append(('test', 'test_df.pkl'))
    for part, filename in selected:
        path = Path(args.data_root) / args.dataset_name / f'1_{args.dataset_name}_Splitted_Datasets' / filename
        result[part] = read_sequences(path, args, mapping)
        if not result[part]:
            raise ValueError(f'{part} split empty')
        for seq in result[part]:
            if seq.sid in existing:
                raise ValueError(f'Node_block_id leakage: {seq.sid} in {existing[seq.sid]} and {part}')
            existing[seq.sid] = part
    print('Validated split sequences:', {k:len(v) for k,v in result.items()}, flush=True)
    print('Original Node_block_id preserved; no new IDs created.', flush=True)
    print('IMPORTANT: original event chronological ordering must be verified upstream.', flush=True)
    return result


def fix_length(vectors, window_size, tail=False):
    out = np.zeros((window_size, 768), np.float32)
    selected = vectors[-window_size:] if tail else vectors[:window_size]
    out[:len(selected)] = selected
    return out


class OriginalDataset(Dataset):
    def __init__(self, sequences, window):
        self.sequences, self.window = sequences, window
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, i):
        seq = self.sequences[i]
        label = np.array([1 - seq.label, seq.label], np.float32)
        return torch.from_numpy(fix_length(seq.events, self.window)), torch.from_numpy(label)


def choose_device(s):
    if s == 'auto':
        s = 'cuda' if torch.cuda.is_available() else 'cpu'
    if s == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    return torch.device(s)


def make_model(args, device):
    return Model(mode=args.mode, num_layers=args.num_layers, adapter_size=args.adapter_size,
                 dim=768, window_size=args.model_window, nhead=8,
                 dim_feedforward=3072, dropout=0.1).to(device)


def checkpoint_path(args, suffix):
    return Path(args.run_dir) / args.dataset_name / suffix


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, title=''):
    training = optimizer is not None
    model.train(training)
    losses, pred, true = [], [], []
    with torch.enable_grad() if training else torch.no_grad():
        for x, y in tqdm(loader, desc=title):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
            losses.append(float(loss.item()))
            pred.extend(logits.argmax(1).detach().cpu().tolist())
            true.extend(y.argmax(1).detach().cpu().tolist())
    return {'loss': float(np.mean(losses)), 'f1': float(f1_score(true, pred, zero_division=0))}


def train(args, splits, device):
    model = make_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(OriginalDataset(splits['train'], args.model_window), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(OriginalDataset(splits['val'], args.model_window), batch_size=args.batch_size, shuffle=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    epochs=args.epochs, steps_per_epoch=len(train_loader))
    out = checkpoint_path(args, 'model_best.pt'); out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0; history=[]
    for i in range(args.epochs):
        tr = run_epoch(model,train_loader,device,criterion,optimizer,scheduler,f'Train {i+1}/{args.epochs}')
        va = run_epoch(model,val_loader,device,criterion,title=f'Validation {i+1}/{args.epochs}')
        history.append({'epoch':i+1,'train':tr,'validation':va})
        print(f'Epoch {i+1}: train F1={tr["f1"]:.4f} val F1={va["f1"]:.4f}', flush=True)
        checkpoint = {'net':model.state_dict(),'epoch':i+1,'validation_f1':va['f1'],
                      'config':{'model_window':args.model_window,'mode':args.mode,
                                'num_layers':args.num_layers,'adapter_size':args.adapter_size,
                                'embedding_source': args.embedding_map or args.embedding_column,
                                'seed': args.seed}}
        torch.save(checkpoint,checkpoint_path(args,'model_last.pt'))
        if va['f1'] > best:
            best = va['f1'];torch.save(checkpoint,out)
    checkpoint_path(args,'training_history.json').write_text(json.dumps(history,indent=2))
    print('Saved best checkpoint:',out)


def load_trained(args, device):
    path = checkpoint_path(args,'model_best.pt')
    if not path.is_file():
        raise FileNotFoundError(f'Model not trained: {path}')
    ck = torch.load(path,map_location='cpu',weights_only=True)  # trusted checkpoint only
    cfg=ck.get('config',{})
    for key, expected in [('model_window',args.model_window),('mode',args.mode),
                          ('num_layers',args.num_layers),('adapter_size',args.adapter_size)]:
        if cfg.get(key) != expected:
            raise ValueError(f'Checkpoint {key}={cfg.get(key)} != requested {expected}; use original settings')
    source = args.embedding_map or args.embedding_column
    if cfg.get('embedding_source') != source:
        raise ValueError('Embedding input differs from training checkpoint; abort')
    model = make_model(args,device)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in ck['net'].items()},strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def batched_probs(model, inputs, device, batch_size):
    if not inputs:
        return np.array([],dtype=float)
    output=[]
    with torch.inference_mode():
        for start in range(0,len(inputs),batch_size):
            arr = np.stack(inputs[start:start+batch_size])
            logits=model(torch.from_numpy(arr).to(device))
            output.extend(torch.softmax(logits,dim=1)[:,1].cpu().tolist())
    return np.asarray(output,dtype=float)


def original_scores(model,seqs,args,device,desc):
    scores=[]
    for start in tqdm(range(0,len(seqs),args.batch_size),desc=desc):
        chunk=seqs[start:start+args.batch_size]
        inp=[fix_length(s.events,args.model_window,tail=False) for s in chunk]
        scores.extend(batched_probs(model,inp,device,args.batch_size).tolist())
    return scores


def online_scores(model,seqs,args,device,desc):
    # Bounded history. Model window fixed at training; if a larger streaming
    # buffer is requested, only the latest model_window observations are scored.
    # No future events ever enter the model.
    all_scores=[]; pending=[]; slots=[]
    with tqdm(total=sum(len(s.events) for s in seqs),desc=desc,unit='events') as progress:
        for i,seq in enumerate(seqs):
            all_scores.append(np.empty(len(seq.events),dtype=float))
            for j in range(len(seq.events)):
                history=seq.events[max(0,j+1-args.online_history):j+1]
                pending.append(fix_length(history,args.model_window,tail=True))
                slots.append((i,j))
                if len(pending)>=args.batch_size:
                    probs=batched_probs(model,pending,device,args.batch_size)
                    for (a,b),p in zip(slots,probs):all_scores[a][b]=p
                    progress.update(len(pending));pending.clear();slots.clear()
        if pending:
            probs=batched_probs(model,pending,device,args.batch_size)
            for (a,b),p in zip(slots,probs):all_scores[a][b]=p
            progress.update(len(pending))
    return all_scores


def alert_time(probs, threshold, persistence):
    run=0
    for i,p in enumerate(probs):
        run=run+1 if p>=threshold else 0
        if run>=persistence:return i+1
    return None


def metrics(sequences, alert_times, fp_weight, fn_weight, delay_weight):
    positive = sum(s.label for s in sequences)
    negative = len(sequences)-positive
    tp=sum(s.label==1 and t is not None for s,t in zip(sequences,alert_times))
    fp=sum(s.label==0 and t is not None for s,t in zip(sequences,alert_times))
    fn=positive-tp; tn=negative-fp
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/positive if positive else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    detected_ratios=[t/len(s.events) for s,t in zip(sequences,alert_times) if s.label==1 and t is not None]
    output={'Test sequences':len(sequences),'Truly anomalous sequences':positive,
            'TP':tp,'TN':tn,'FP':fp,'FN':fn,'Precision':precision,'Recall':recall,
            'F1':f1,'FPR':fp/negative if negative else 0.,
            'AvgRatio (detected only)':float(np.mean(detected_ratios)) if detected_ratios else None}
    for cutoff in (25,50,75):
        detected=sum(s.label==1 and t is not None and t/len(s.events)<=cutoff/100
                     for s,t in zip(sequences,alert_times))
        output[f'EDR@{cutoff}']=detected/positive if positive else 0.0
    output['FP cost']=fp_weight*fp
    output['FN cost']=fn_weight*fn
    output['Delay cost']=delay_weight*sum(detected_ratios)
    output['Total cost']=output['FP cost']+output['FN cost']+output['Delay cost']
    return output


def calibrate(seqs,scores,args):
    # Only validation labels are consulted for selection. Never test labels.
    candidates=[]
    for threshold in args.thresholds:
        for persistence in args.consecutive_candidates:
            alert=[alert_time(arr,threshold,persistence) for arr in scores]
            m=metrics(seqs,alert,args.fp_weight,args.fn_weight,args.delay_weight)
            candidates.append((threshold,persistence,m))
    allowed=[c for c in candidates if c[2]['FPR']<=args.max_validation_fpr]
    if not allowed:
        print('WARNING: no validation policy meets FPR cap; choose lowest-FPR then F1.',flush=True)
        return max(candidates,key=lambda c:(-c[2]['FPR'],c[2]['F1'],-c[0]))
    return max(allowed,key=lambda c:(c[2]['F1'],-c[2]['FPR'],c[0]))


def evaluate(args,splits,device):
    model=load_trained(args,device)
    # Original branch: same trained fixed-length model using FIRST N tokens,
    # fixed threshold=0.5, and only emits prediction at the end.
    print('Original full-sequence inference (test):',flush=True)
    full_probs=original_scores(model,splits['test'],args,device,'Original test')
    original_times=[len(s.events) if p>=0.5 else None for s,p in zip(splits['test'],full_probs)]
    original=metrics(splits['test'],original_times,args.fp_weight,args.fn_weight,args.delay_weight)
    print('Online validation inference:',flush=True)
    validation_scores=online_scores(model,splits['val'],args,device,'Online validation')
    threshold,persistence,val_metric=calibrate(splits['val'],validation_scores,args)
    print(f'Calibrated on validation only: threshold={threshold}, consecutive={persistence}, '
          f'validation F1={val_metric["F1"]:.4f}, FPR={val_metric["FPR"]:.4f}',flush=True)
    print('Online test inference:',flush=True)
    test_scores=online_scores(model,splits['test'],args,device,'Online test')
    online_times=[alert_time(arr,threshold,persistence) for arr in test_scores]
    online=metrics(splits['test'],online_times,args.fp_weight,args.fn_weight,args.delay_weight)
    out=checkpoint_path(args,'comparison.csv');out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key in original:
        rows.append({'Metric':key,'Original LogFormer':original[key],'Online LogFormer':online[key]})
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['Metric','Original LogFormer','Online LogFormer']);writer.writeheader();writer.writerows(rows)
    payload={'dataset':args.dataset_name,'original':original,'online':online,
             'alert_policy':{'validation_threshold':threshold,'consecutive':persistence,
                             'max_validation_fpr':args.max_validation_fpr},
             'config':{'model_window':args.model_window,'online_history':args.online_history,
                       'label_column':args.label_column,'embedding_source':args.embedding_map or args.embedding_column,
                       'fp_weight':args.fp_weight,'fn_weight':args.fn_weight,'delay_weight':args.delay_weight}}
    checkpoint_path(args,'comparison.json').write_text(json.dumps(payload,indent=2))
    widths=[35,22,22]
    print('\n'+' '.join(k.ljust(w) for k,w in zip(['Metric','Original LogFormer','Online LogFormer'],widths)))
    print('-'*83)
    for r in rows:
        def fmt(v):return 'N/A' if v is None else f'{v:.6f}' if isinstance(v,float) else str(v)
        print(f'{r["Metric"]:<35} {fmt(r["Original LogFormer"]):>22} {fmt(r["Online LogFormer"]):>22}')
    print('\nSaved:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_name',required=True)
    p.add_argument('--data_root',default='../../LWADLS/datasets')
    p.add_argument('--mode_run',choices=['train','evaluate','all','audit'],default='all')
    p.add_argument('--id-column',default='Node_block_id')
    p.add_argument('--event-column',default='EventId')
    p.add_argument('--label-column',default='Original_Label')
    p.add_argument('--embedding-column',default='EventEmbedding',help='PKL column containing original 768-D vectors')
    p.add_argument('--embedding-map',default=None,help='NPZ with event_ids + embeddings from ORIGINAL LogFormer feature encoder')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--run-dir',default='logformer_runs')
    p.add_argument('--model-window',type=int,default=120,help='Training-time fixed classifier length; MUST match checkpoint')
    p.add_argument('--online-history',type=int,default=120,help='Maximum history accessible at online inference')
    p.add_argument('--num-layers',type=int,default=1)
    p.add_argument('--adapter-size',type=int,default=64)
    p.add_argument('--mode',choices=['classifier','adapter'],default='classifier')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--thresholds',type=float,nargs='+',default=[0.5,0.6,0.7,0.8,0.9,0.95,0.99])
    p.add_argument('--consecutive-candidates',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--max-validation-fpr',type=float,default=0.05)
    p.add_argument('--fp-weight',type=float,default=1.0)
    p.add_argument('--fn-weight',type=float,default=1.0)
    p.add_argument('--delay-weight',type=float,default=1.0)
    args=p.parse_args()
    if args.model_window<=0 or args.online_history<=0:
        p.error('model-window and online-history must be positive')
    if not args.embedding_map:
        print('Using PKL per-event vector column:',args.embedding_column,flush=True)
    set_seed(args.seed)
    # Never load the test PKL during training or checkpoint selection.
    splits=load_splits(args, include_test=args.mode_run in ('evaluate','audit'))
    if args.mode_run=='audit':return
    device=choose_device(args.device)
    print('Selected device:',device,flush=True)
    if args.mode_run in ('train','all'):train(args,splits,device)
    if args.mode_run=='all':
        # Re-open all splits after training, now including held-out test, to
        # verify original ID disjointness before final evaluation.
        splits=load_splits(args, include_test=True)
    if args.mode_run in ('evaluate','all'):evaluate(args,splits,device)


if __name__=='__main__':
    main()
