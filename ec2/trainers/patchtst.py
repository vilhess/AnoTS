import sys
sys.path.append("/home/svilhes/Bureau/these/AnoTS/ec2/")
# sys.path.append("/Users/samyvilhes/Desktop/these/AnoTS/ec2") on personal mac

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from types import SimpleNamespace

from dataset import get_datasets
from models.patchtst.patchtst import Model

DEVICE="cuda"
BATCH_SIZE=128
EPOCHS=100

CONTEXT_WINDOW=10
TARGET_WINDOW=1

REVIN=False
ext = "_rev" if REVIN else ""

trainset, valset, testset, dataset = get_datasets(window=CONTEXT_WINDOW, lbl_as_feat=False)

args = {
    "seq_len":CONTEXT_WINDOW,
    "pred_len":TARGET_WINDOW,
    "enc_in":1,
    "e_layers":3,
    "n_heads":16,
    "d_model":16,
    "d_ff":256,
    "dropout":0.3,
    "fc_dropout":0.3,
    "head_dropout":0,
    "patch_len":16,
    "stride":8,
    "individual":1,
    "padding_patch":"end",
    "revin":int(REVIN),
    "affine":1,
    "subtract_last":0,
    "decomposition":1,
    "kernel_size":25
    }

args = SimpleNamespace(**args)
model = Model(configs=args).to(DEVICE)

LR=1e-4
PCT_START = 0.3


trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer,
                                    steps_per_epoch = len(trainloader),
                                    pct_start = PCT_START,
                                    epochs = EPOCHS,
                                    max_lr = LR)

pbar = trange(EPOCHS)

for epoch in pbar:
    epoch_loss = 0

    for x_cont, _, target, _ in trainloader:
        x_cont, target = x_cont[:, :, 0].unsqueeze(2).to(DEVICE), target.unsqueeze(1).to(DEVICE)
        preds = model(x_cont)
        preds = preds.squeeze(2)
        loss = criterion(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss+=loss.item()
    pbar.set_description(f"for epoch {epoch+1}/{EPOCHS} ; loss : {loss.item()}")

checkpoint = {
    'model':model.state_dict(),
    'args':args
    }
torch.save(checkpoint, f'checkpoints/patchtst{ext}.pkl')

valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

criterion_wise = nn.MSELoss(reduction="none")

val_errors = []

model.eval()
with torch.no_grad():
    for x_cont, _, target, _ in valloader:
            x_cont, target = x_cont[:, :, 0].unsqueeze(2).to(DEVICE), target.unsqueeze(1).to(DEVICE)
            val_preds = model(x_cont)
            val_preds = val_preds.squeeze(2)
            errors = criterion_wise(target, val_preds)
            val_errors.append(errors)
val_errors = torch.cat(val_errors).cpu()
val_scores = - val_errors

test_errors = []

model.eval()
with torch.no_grad():
    for x_cont, _, target, _ in testloader:
            x_cont, target = x_cont[:, :, 0].unsqueeze(2).to(DEVICE), target.unsqueeze(1).to(DEVICE)
            test_preds = model(x_cont)
            test_preds = test_preds.squeeze(2)
            errors = criterion_wise(target, test_preds)
            test_errors.append(errors)
test_errors = torch.cat(test_errors).cpu()
test_scores = - test_errors
test_p_values = (1 + torch.sum(test_scores >= val_scores.squeeze(1), dim=1)) / (len(val_scores.squeeze(1)) + 1)

np.save(f'pvalues/patchtst{ext}.npy', test_p_values.numpy())
# Visu
train_indices = dataset.train_indices
val_indices = dataset.val_indices
test_indices = dataset.test_indices
traindf = dataset.dataframe[dataset.dataframe["timestamp"].isin(train_indices)]
valdf = dataset.dataframe[dataset.dataframe["timestamp"].isin(val_indices)]
testdf = dataset.dataframe[dataset.dataframe["timestamp"].isin(test_indices)]

anomaly_periods = dataset.anomaly_periods
traindf.index = pd.to_datetime(train_indices)
valdf.index = pd.to_datetime(val_indices)
testdf.index = pd.to_datetime(test_indices)
plt.figure(figsize=(20, 10))

#plt.plot(traindf.index, traindf['value'], label="Train", c="blue")
plt.plot(valdf.index, valdf['value'], label="Val", c="green")
plt.plot(testdf.index, testdf['value'], label="Test", c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Time Series with Anomaly Periods - After Normalize', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.grid(True)

plt.savefig('figures/valtest.png')
plt.close()


plt.figure(figsize=(20, 10))

#plt.plot(traindf.index, traindf['value'], label="Train", c="blue")
plt.plot(valdf.index, val_errors.cpu(), label="Val", c="green")
plt.plot(testdf.index, test_errors.cpu(), label="Test", c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Prediction errors with Anomaly Periods - After Normalize', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Squared error', fontsize=14)
plt.grid(True)

plt.savefig(f'figures/patchtst/prederrors{ext}.png')
plt.close()
