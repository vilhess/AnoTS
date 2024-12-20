import sys
sys.path.append("/home/svilhes/Bureau/these/nyctaxi/")

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataset import get_datasets
from models.tranad import TranAD, CatEmbedding

DEVICE="cuda"
WINDOW=10
BATCH_SIZE=128
LR=1e-3
EPOCHS=5

trainset, valset, testset, dataset = get_datasets(window=WINDOW, lbl_as_feat=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

model = TranAD(in_dim=8, window=WINDOW+1).to(DEVICE)
embedder = CatEmbedding().to(DEVICE)

optimizer_model = torch.optim.AdamW(model.parameters() , lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, 5, 0.9)

optimizer_embed = torch.optim.Adam(embedder.parameters(), lr=1e-4)

criterion = nn.MSELoss(reduction="none")

pbar = trange(EPOCHS)
for epoch in pbar:
    epoch_loss=0

    for i, (x_cont, x_cat, _, _) in enumerate(trainloader):
        x_cont = x_cont.to(DEVICE).permute(1, 0, 2)
        x_cat = x_cat.to(DEVICE).permute(1, 0, 2)
        e = embedder(x_cat)
        x = torch.cat([x_cont, e], dim=2)
        elem = x[-1, :, :].view(1, x.size(1), x.size(2))
        x1, x2 = model(x, elem)
        loss = 1/(epoch+1) * criterion(elem, x1) + (1 - 1/(epoch+1)) * criterion(elem, x2)
        loss = torch.mean(loss)
        optimizer_model.zero_grad()
        optimizer_embed.zero_grad()
        loss.backward()
        optimizer_model.step()
        optimizer_embed.step()
        epoch_loss+=loss.item()
    pbar.set_description(f"Training loss: {epoch_loss/len(trainloader)}")
    scheduler.step()

model.eval()
embedder.eval()

checkpoint = {"model":model.state_dict(), "embed":embedder.state_dict()}
torch.save(checkpoint, 'checkpoints/tranad.pkl')

valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

val_scores = []

for x_cont, x_cat, _, _ in valloader:
    x_cont = x_cont.to(DEVICE).permute(1, 0, 2)
    x_cat = x_cat.to(DEVICE).permute(1, 0, 2)
    with torch.no_grad():
        e = embedder(x_cat)
        x = torch.cat([x_cont, e], dim=2)
        elem = x[-1, :, :].view(1, x.size(1), x.size(2))
        _, pred = model(x, elem)

    loss = criterion(pred, elem).permute(1, 0, 2)
    loss = torch.mean(loss, dim=(1, 2))
    val_scores.append(-loss)

val_scores = torch.cat(val_scores)

test_scores = []

for x_cont, x_cat, _, _ in testloader:
    x_cont = x_cont.to(DEVICE).permute(1, 0, 2)
    x_cat = x_cat.to(DEVICE).permute(1, 0, 2)
    with torch.no_grad():
        e = embedder(x_cat)
        x = torch.cat([x_cont, e], dim=2)
        elem = x[-1, :, :].view(1, x.size(1), x.size(2))
        _, pred = model(x, elem)

    loss = criterion(pred, elem).permute(1, 0, 2)
    loss = torch.mean(loss, dim=(1, 2))
    test_scores.append(-loss)

test_scores = torch.cat(test_scores)
test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores, dim=1)) / (len(val_scores) + 1)
test_p_values = test_p_values.cpu()

np.save('pvalues/tranad.npy', test_p_values.numpy())

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
plt.plot(valdf.index, val_scores.cpu(), label="Val", c="green")
plt.plot(testdf.index, test_scores.cpu(), label="Test", c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Normality Score for each timestamps', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Normality Score', fontsize=14)
plt.grid(True)

plt.savefig('figures/tranad/prederrors.png')