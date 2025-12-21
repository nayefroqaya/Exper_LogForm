import os
import argparse
import random
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm

from dataloader import DataGenerator
from model import Model

# =========================
# Arguments
# =========================
parser = argparse.ArgumentParser()
# fine-tuning setting
parser.add_argument('--pretrained_log_name', type=str,
                    default='HDFS', help='log file name')
parser.add_argument("--load_path", type=str,
                    default='checkpoints/train_HDFS_classifier_1_64_5e-05-best.pt', help="latest model path")
parser.add_argument('--log_name', type=str,
                    default='BGL', help='log file name')
parser.add_argument('--tune_mode', type=str, default='adapter',
                    help='tune adapter or classifier only')
# model setting
parser.add_argument('--num_layers', type=int, default=1,
                    help='num of encoder layer')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--window_size', type=int,
                    default='20', help='log sequence length')
parser.add_argument('--adapter_size', type=int, default=64,
                    help='adapter size')
parser.add_argument('--epoch', type=int, default=20,
                    help='epoch')
args = parser.parse_args()

# =========================
# Folders
# =========================
os.makedirs("result", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
suffix = f'{args.log_name}_from_{args.pretrained_log_name}_{args.tune_mode}_{args.num_layers}_{args.adapter_size}_{args.lr}_{args.epoch}'

with open(f'result/tune_{suffix}.txt', 'w', encoding='utf-8') as f:
    f.write(str(args) + '\n')

# =========================
# Hyperparameters
# =========================
EMBEDDING_DIM = 768
batch_size = 64
epochs = args.epoch

# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Fix seeds
# =========================
warnings.filterwarnings('ignore')
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True

# =========================
# Load preprocessed data
# =========================
train_path = f'./preprocess/preprocessed_data/{args.log_name}_training_block_w{args.window_size}.npz'
test_path = f'./preprocess/preprocessed_data/{args.log_name}_testing_block_w{args.window_size}.npz'
train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)

x_train, y_train = train_data['x'], train_data['y']
x_test, y_test = test_data['x'], test_data['y']
del train_data, test_data

# =========================
# DataLoaders
# =========================
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.float32)),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                   torch.tensor(y_test, dtype=torch.float32)),
    batch_size=batch_size, shuffle=False
)

# =========================
# Model
# =========================
model = Model(
    mode='adapter',
    num_layers=args.num_layers,
    adapter_size=args.adapter_size,
    dim=EMBEDDING_DIM,
    window_size=args.window_size,
    nhead=8,
    dim_feedforward=4*EMBEDDING_DIM,
    dropout=0.1
)

# =========================
# Fine-tuning mode
# =========================
if args.tune_mode == 'adapter':
    model.train_adapter()
elif args.tune_mode == 'classifier':
    model.train_classifier()
elif args.tune_mode == 'tuning':
    for param in model.parameters():
        param.requires_grad = True

# =========================
# Load pretrained weights
# =========================
if args.pretrained_log_name != 'random':
    checkpoint = torch.load(args.load_path, map_location='cpu')
    net = checkpoint['net']
    # remove fc layer weights if shapes differ
    net.pop('module.fc1.weight', None)
    net.pop('module.fc1.bias', None)
    r = model.load_state_dict(net, strict=False)
    with open(f'result/tune_{suffix}.txt', 'a', encoding='utf-8') as f:
        f.write(f'Loaded pretrained model {args.load_path}\nLoad result: {r}\n')

# =========================
# Device and DataParallel
# =========================
if device.type == "cuda":
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    if gpu_count > 1:
        model = nn.DataParallel(model)
model = model.to(device)

# =========================
# Optimizer & Scheduler
# =========================
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr, epochs=epochs, steps_per_epoch=len(train_loader)
)
criterion = nn.BCEWithLogitsLoss()

# =========================
# Training loop
# =========================
best_f1 = 0
log_interval = 100

for epoch in range(epochs):
    model.train()
    loss_all, train_pred, train_true = [], [], []
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        loss_all.append(loss.item())
        train_pred.extend(out.argmax(1).tolist())
        train_true.extend(y.argmax(1).tolist())

        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_f1 = f1_score(train_true, train_pred)
            time_cost = time.time() - start_time
            print(f"| epoch {epoch} | batch {batch_idx} | loss {np.mean(loss_all):.5f} | f1 {cur_f1:.5f} | time {time_cost:.2f}s | lr {scheduler.get_last_lr()}")
            with open(f'result/tune_{suffix}.txt', 'a', encoding='utf-8') as f:
                f.write(f"| epoch {epoch} | batch {batch_idx} | loss {np.mean(loss_all):.5f} | f1 {cur_f1:.5f} | time {time_cost:.2f}s | lr {scheduler.get_last_lr()}\n")
            loss_all, train_pred, train_true = [], [], []
            start_time = time.time()

    # =========================
    # Evaluation
    # =========================
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x).cpu()
            y_pred_list.append(out)
            y_true_list.append(y)

    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average='binary')

    print(f"Epoch {epoch}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    with open(f'result/tune_{suffix}.txt', 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")

    # =========================
    # Save checkpoints
    # =========================
    checkpoint = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    if f1 > best_f1:
        best_f1 = f1
        torch.save(checkpoint, f'checkpoints/tune_{suffix}-best.pt')
    torch.save(checkpoint, f'checkpoints/tune_{suffix}-latest.pt')
