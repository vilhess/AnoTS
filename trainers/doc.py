import sys
sys.path.append("/home/svilhes/Bureau/these/nyctaxi/")

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataset import get_datasets
from models.doc import DOC

DEVICE="cuda"
WINDOW=32
BATCH_SIZE=128
LR=1e-4
EPOCHS=150
WEIGHT_DECAY=1e-6
LATENT_DIM=32

trainset, valset, testset, dataset = get_datasets(window=WINDOW, lbl_as_feat=True)

model = DOC(num_cont_var=4, embedding_dim=2, hidden_size=128, seq_len=WINDOW+1, latent_dim=LATENT_DIM, num_layers=2, bidirectional=True).to(DEVICE)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0], gamma=0.1)

# Init center
model.eval()

n_samples = 0
eps=0.1
c = torch.zeros(LATENT_DIM).to(DEVICE)

model.eval()
with torch.no_grad():
    for x_cont, x_cat, target, anomaly in trainloader:
        x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
        proj = model(x_cont, x_cat)
        n_samples += proj.shape[0]
        c += torch.sum(proj, dim=0)
c /= n_samples

c[(abs(c) < eps) & (c < 0)] = -eps
c[(abs(c) < eps) & (c > 0)] = eps

model.train()

pbar = trange(EPOCHS, desc="Training")
for epoch in pbar:
        
    curr_loss = 0
    for x_cont, x_cat, target, anomaly in trainloader:
        x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
        optimizer.zero_grad()
        proj = model(x_cont, x_cat)
        dist = torch.sum((proj - c) ** 2, dim=1)
        loss = torch.mean(dist)
        curr_loss+=loss.item()
            
        loss.backward()
        optimizer.step()

    scheduler.step()
    pbar.set_description(f"For epoch {epoch+1}/{EPOCHS} ; loss : {curr_loss/len(trainloader)}")
    

checkpoint = {"state_dict":model.state_dict(), 'center':c.detach().cpu()}
torch.save(checkpoint, 'checkpoints/doc.pkl')

valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

val_proj = []

model.eval()
with torch.no_grad():
    for x_cont, x_cat, target, anomaly in valloader:
        x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
        proj = model(x_cont, x_cat)
        val_proj.append(proj)
val_proj = torch.cat(val_proj)

val_dist = torch.sum((val_proj - c) ** 2, dim=1)
val_scores = - val_dist

test_proj = []

model.eval()
with torch.no_grad():
    for x_cont, x_cat, target, anomaly in testloader:
        x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
        proj = model(x_cont, x_cat)
        test_proj.append(proj)
test_proj = torch.cat(test_proj)

test_dist = torch.sum((test_proj - c) ** 2, dim=1)
test_scores = - test_dist

test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores, dim=1)) / (len(val_scores) + 1)
test_p_values = test_p_values.cpu()

np.save('pvalues/doc.npy', test_p_values.numpy())

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
plt.plot(valdf.index, val_dist.cpu(), label="Val", c="green")
plt.plot(testdf.index, test_dist.cpu(), label="Test", c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Projection distance to center with Anomaly Periods - After Normalize', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Squared error', fontsize=14)
plt.grid(True)

plt.savefig('figures/doc/prederrors.png')