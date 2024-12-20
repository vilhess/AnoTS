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
from models.aelstm import AELSTM

DEVICE="cuda"

WINDOW=32
BATCH_SIZE=128
LR=1e-4
EPOCHS=150

trainset, valset, testset, dataset = get_datasets(window=WINDOW, lbl_as_feat=True)

model = AELSTM(num_cont_var=2, embedding_dim=2, hidden_size=128, latent_dim=64).to(DEVICE)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()

pbar = trange(EPOCHS, desc="Training")
for epoch in pbar:
        
    curr_loss = 0
    for x_cont, x_cat, target, anomaly in trainloader:
        x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
        optimizer.zero_grad()
        targets, preds = model(x_cont, x_cat)
        loss = criterion(preds, targets)
        curr_loss+=loss.item()
            
        loss.backward()
        optimizer.step()

    pbar.set_description(f"For epoch {epoch+1}/{EPOCHS} ; loss : {curr_loss/len(trainloader)}")
    

checkpoint = {"state_dict":model.state_dict()}
torch.save(checkpoint, 'checkpoints/aelstm.pkl')

valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

val_errors = []

model.eval()
with torch.no_grad():
    for x_cont, x_cat, target, anomaly in valloader:
            x_cont, x_cat, target = x_cont.to(DEVICE), x_cat.to(DEVICE), target.unsqueeze(1).to(DEVICE)
            targets, preds = model(x_cont, x_cat)
            errors = torch.abs(targets - preds).sum(dim=1)
            val_errors.append(errors)
val_errors = torch.cat(val_errors).cpu()
val_scores = - val_errors

test_errors = []

model.eval()
with torch.no_grad():
    for x_cont, x_cat, target, anomaly in testloader:
            x_cont, x_cat, target = x_cont.to(DEVICE), x_cat.to(DEVICE), target.unsqueeze(1).to(DEVICE)
            targets, preds = model(x_cont, x_cat)
            errors = torch.abs(targets - preds).sum(dim=1)
            test_errors.append(errors)
test_errors = torch.cat(test_errors).cpu()
test_scores = - test_errors

test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores, dim=1)) / (len(val_scores) + 1)
test_p_values = test_p_values.cpu()

np.save('pvalues/aelstm.npy', test_p_values.numpy())

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

plt.plot(traindf.index, traindf['value'], label="Train", c="blue")
plt.plot(valdf.index, val_errors.cpu(), label="Val", c="green")
plt.plot(testdf.index, test_errors.cpu(), label="Test", c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Reconstruction Error with Anomaly Periods - After Normalize', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Squared error', fontsize=14)
plt.grid(True)

plt.savefig('figures/aelstm/prederrors.png')
plt.close()