import argparse
import time
import os
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

parser = argparse.ArgumentParser()
parser.add_argument('--log_name', type=str,
                    default='BGL', help='log file name')
parser.add_argument('--window_size', type=int,
                    default='120', help='log sequence length')
parser.add_argument('--mode', type=str, default='classifier',
                    help='use adapter or not')
parser.add_argument('--num_layers', type=int, default=1,
                    help='num of encoder layer')
parser.add_argument('--adapter_size', type=int, default=64,
                    help='adapter size')
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument("--resume", type=int, default=0,
                    help="resume training of model (0/no, 1/yes)")
parser.add_argument("--load_path", type=str,
                    default='checkpoints/model-latest.pt', help="latest model path")
parser.add_argument('--preprocessed_dir', type=str, default='preprocess/preprocessed_data',
                    help='directory containing BGL_training_block_w*.npz and BGL_testing_block_w*.npz')
args = parser.parse_args()
suffix = f'{args.log_name}_{args.mode}_{args.num_layers}_{args.adapter_size}_{args.lr}'
os.makedirs('result', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
with open(f'result/train_{suffix}.txt', 'a', encoding='utf-8') as f:
    f.write(str(args)+'\n')

# hyper-parameters
EMBEDDING_DIM = 768
batch_size = 64
epochs = 5
lr = args.lr
# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# fix all random seeds
warnings.filterwarnings('ignore')
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

# load data Hdfs
# =========================
# Load preprocessed data
# =========================
train_path = os.path.join(args.preprocessed_dir, f'{args.log_name}_training_block_w{args.window_size}.npz')
test_path = os.path.join(args.preprocessed_dir, f'{args.log_name}_testing_block_w{args.window_size}.npz')
train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)


x_train, y_train = train_data['x'], train_data['y']
x_test, y_test = test_data['x'], test_data['y']
del train_data
del test_data

print(f"Loaded train x shape: {x_train.shape}, y shape: {y_train.shape}")
print(f"Loaded test  x shape: {x_test.shape}, y shape: {y_test.shape}")

if len(x_train) == 0:
    raise ValueError(
        f"Training NPZ is empty: {train_path}. "
        "Rerun preprocessing with block_length_mode: keep_variable or pad_truncate."
    )

if len(x_test) == 0:
    raise ValueError(
        f"Testing NPZ is empty: {test_path}. "
        "Rerun preprocessing with block_length_mode: keep_variable or pad_truncate."
    )

train_generator = DataGenerator(x_train, y_train, args.window_size)
test_generator = DataGenerator(x_test, y_test, args.window_size)
train_loader = torch.utils.data.DataLoader(
    train_generator, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_generator, batch_size=batch_size, shuffle=False)



# automatically choose GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# if multiple GPUs, get all device IDs
if torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))
    print("Using multiple GPUs:", device_ids)
else:
    device_ids = None


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

# move to device
model = model.to(device)

# wrap with DataParallel **only if multiple GPUs are available**
if device_ids and len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.7, patience=4, threshold=1e-4, verbose=True)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
criterion = nn.BCEWithLogitsLoss()

start_epoch = -1
if args.resume == 1:
    path_checkpoint = args.load_path
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print("resume training from epoch ", start_epoch)


best_f1 = 0
log_interval = 100
total_training_start_time = time.time()
for epoch in range(start_epoch+1, epochs):
    loss_all, f1_all = [], []
    train_loss = 0
    train_pred, train_true = [], []

    model.train()
    start_time = time.time()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        x, y = data[0].to(device), data[1].to(device)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_pred.extend(out.argmax(1).tolist())
        train_true.extend(y.argmax(1).tolist())

        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_loss = train_loss / log_interval
            # scheduler.step(cur_loss)
            cur_f1 = f1_score(train_true, train_pred)
            time_cost = time.time()-start_time

            with open(f'result/train_{suffix}.txt', 'a', encoding='utf-8') as f:
                f.write(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                        f'loss {cur_loss:2.5f} |'
                        f'f1 {cur_f1:.5f} |'
                        f'time {time_cost:4.2f} |'
                        f'lr {scheduler.get_last_lr()}\n')
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                  f'loss {cur_loss} |'
                  f'f1 {cur_f1}',
                  f'lr {scheduler.get_last_lr()}')

            loss_all.append(train_loss)
            f1_all.append(cur_f1)

            start_time = time.time()
            train_loss = 0
            train_acc = 0

    train_loss = float(np.mean(loss_all)) if loss_all else train_loss / max(1, len(train_loader))
    print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, train_loss))

    model.eval()
    y_pred_list, y_true_list = [], []
    test_start_time = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            x, y = data[0].to(device), data[1].to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            out = model(x).cpu()
            y_pred_list.append(out)
            y_true_list.append(y.cpu())

    test_prediction_time = time.time() - test_start_time

    if len(y_pred_list) == 0:
        raise ValueError(
            "test_loader is empty. The testing NPZ has zero samples. "
            "Rerun preprocessing with block_length_mode: keep_variable or pad_truncate."
        )

    # calculate metrics
    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    report = precision_recall_fscore_support(y_true, y_pred, average='binary')
    with open(f'result/train_{suffix}.txt', 'a', encoding='utf-8') as f:
        f.write('number of epochs:'+str(epoch)+'\n')
        f.write('Number of testing data:'+str(x_test.shape[0])+'\n')
        f.write('Precision:'+str(report[0])+'\n')
        f.write('Recall:'+str(report[1])+'\n')
        f.write('F1 score:'+str(report[2])+'\n')
        f.write('Training time so far:'+str(time.time() - total_training_start_time)+'\n')
        f.write('Test prediction time:'+str(test_prediction_time)+'\n')
        f.write('all_loss:'+str(loss_all)+'\n')
        f.write('\n')
        f.close()

    print(f'Number of testing data: {x_test.shape[0]}')
    print(f'Precision: {report[0]:.4f}')
    print(f'Recall: {report[1]:.4f}')
    print(f'F1 score: {report[2]:.4f}')
    print(f'Training time so far: {time.time() - total_training_start_time:.4f}s')
    print(f'Test prediction time: {test_prediction_time:.4f}s')

    ckpt_path = 'checkpoints/'
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }
    if report[2] > best_f1:
        best_f1 = report[2]
        torch.save(checkpoint, os.path.join(
            ckpt_path, f'train_{suffix}-best.pt'))
    torch.save(checkpoint, os.path.join(
        ckpt_path, f'train_{suffix}-latest.pt'))