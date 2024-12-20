import sys
sys.path.append("/home/svilhes/Bureau/these/ec2/")

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataset import get_datasets
from models.drocc import DROCCTrainer, LSTM, adjust_learning_rate

DEVICE="cuda"
WINDOW=32
BATCH_SIZE=128

LR=1e-4
EPOCHS=40
RADIUS=3
GAMMA=2
LAMBDA=1
ONLY_CE_EPOCHS=5

ascent_step_size=1e-3
ascent_num_steps=50

trainset, valset, testset, dataset = get_datasets(window=WINDOW, lbl_as_feat=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)

model = LSTM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainer = DROCCTrainer(model=model, optimizer=optimizer, lamda=LAMBDA, radius=RADIUS, gamma=GAMMA, device=DEVICE)

trainer.train(train_loader=trainloader, learning_rate=LR, lr_scheduler=adjust_learning_rate, total_epochs=EPOCHS, only_ce_epochs=ONLY_CE_EPOCHS, ascent_step_size=ascent_step_size, ascent_num_steps=ascent_num_steps)

checkpoint = trainer.model.state_dict()
torch.save(checkpoint, "checkpoints/drocc.pkl")

model = trainer.model

valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

val_logits = []

model.eval()
with torch.no_grad():
    for x_cont, x_cat, _, _ in valloader:
        x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
        logits, _ = model(x_cont, x_cat)
        val_logits.append(torch.sigmoid(logits))
val_logits = torch.cat(val_logits).cpu()

test_logits = []

model.eval()
with torch.no_grad():
    for x_cont, x_cat, _, _ in testloader:
        x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
        logits, _ = model(x_cont, x_cat)
        test_logits.append(torch.sigmoid(logits))
test_logits = torch.cat(test_logits).cpu()

test_p_values = (1 + torch.sum(test_logits.unsqueeze(1) >= val_logits, dim=1)) / (len(val_logits) + 1)
test_p_values = test_p_values.cpu()

np.save('pvalues/drocc.npy', test_p_values.numpy())

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
plt.plot(valdf.index, val_logits.cpu(), label="Val", c="green")
plt.plot(testdf.index, test_logits.cpu(), label="Test", c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Probabiliy of being normal', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.grid(True)

plt.savefig('figures/drocc/prederrors.png')
