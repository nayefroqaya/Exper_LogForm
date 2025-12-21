import os
import argparse
import time
import random
import warnings
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
parser.add_argument('--log_name', type=str, default='BGL', help='log file name')
parser.add_argument('--window_size', type=int, default=120, help='log sequence length')
parser.add_argument('--mode', type=str, default='classifier', help='model mode')
parser.add_argument('--num_layers', type=int, default=1, help='num of encoder layer')
parser.add_argument('--adapter_size', type=int, default=64, help='adapter size')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument("--resume", type=int, default=0, help="resume training of model (0=no, 1=yes)")
parser.add_argument("--load_path", type=str, default='checkpoints/model-latest.pt', help="latest model path")
args = parser.parse_args()

# =========================
# Create folders if they don't exist
# =========================
os.makedirs("result", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

suffix = f'{args.log_name}_{args.mode}_{args.num_layers}_{args.adapter_size}_{args.lr}'
with open(f'result/train_{suffix}.txt', 'a', encoding='utf-8') as f:
    f.write(str(args)+'\n')

# =========================
# Hyperparameters
# =========================
EMBEDDING_DIM = 768
batch_size = 64
epochs = 5
lr = args.lr

# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Load preprocessed data
# =========================

training_data = np.load(
    f'./preprocess/preprocessed_data/{args.log_name}_training_block_w{args.window_size}.npz', allow_pickle=True)
testing_data = np.load(
    f'./preprocess/preprocessed_data/{args.log_name}_testing_block_w{args.window_size}.npz', allow_pickle=True)


x_train, y_train = training_data['x'], training_data['y']
x_test, y_test = testing_data['x'], testing_data['y']

del training_data, testing_data

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
    mode=args.mode,
    num_layers=args.num_layers,
    adapter_size=args.adapter_size,
    dim=EMBEDDING_DIM,
    window_size=args.window_size,
    nhead=8,
    dim_feedforward=4*EMBEDDING_DIM,
    dropout=0.1
)

# Automatically use multiple GPUs if available
if device.type == "cuda":
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    if gpu_count > 1:
        print("Using DataParallel on all GPUs")
        model = nn.DataParallel(model)

# Move model to device
model = model.to(device)

# =========================
# Optimizer & Scheduler
# =========================
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
criterion = nn.BCEWithLogitsLoss()

# =========================
# Training loop (simplified)
# =========================
best_f1 = 0
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            out = model(x_batch).cpu()
            y_pred_list.append(out)
            y_true_list.append(y_batch)

    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    y_true = torch.cat(y_true_list, dim=0).numpy()

    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average='binary')

    print(f"Epoch {epoch}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Save best model
    checkpoint = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    if f1 > best_f1:
        best_f1 = f1
        torch.save(checkpoint, f'checkpoints/train_{suffix}-best.pt')
    torch.save(checkpoint, f'checkpoints/train_{suffix}-latest.pt')