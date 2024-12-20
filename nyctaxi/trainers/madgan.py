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
from models.madgan import Discriminator, Generator, get_best_latent

# These parameteres work well
DEVICE="cuda"
BATCH_SIZE=256
LR=1e-4
EPOCHS=100

WINDOW=4
LATENT_DIM=32
HIDDEN_DIM=256
NUM_LAYERS=2
BIDIRECTIONAL=False

trainset, valset, testset, dataset = get_datasets(window=WINDOW, lbl_as_feat=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

disc = Discriminator(num_cont_var=2, hidden_dim=int(HIDDEN_DIM/2), num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL).to(DEVICE)
gen = Generator(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_cont_var=2, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL).to(DEVICE)

optim_disc = optim.Adam(disc.parameters(), lr=LR)
optim_gen = optim.Adam(gen.parameters(), lr=LR)

pbar = trange(EPOCHS)
for epoch in pbar:
    epoch_gen_loss=0
    epoch_disc_loss=0

    for i, (x_cont, _, _, _) in enumerate(trainloader):
        bs = x_cont.size(0)
        x_cont = x_cont[:, :, :-2]
        x_cont = x_cont.to(DEVICE)

        optim_disc.zero_grad()
        
        ones = torch.ones(bs, 1).to(DEVICE)
        zeros = torch.zeros(bs, 1).to(DEVICE)

        pred_disc_true = disc(x_cont)
        loss_disc_true = F.binary_cross_entropy_with_logits(pred_disc_true, zeros)

        z = torch.randn((bs, WINDOW+1, LATENT_DIM)).to(DEVICE)
    
        fake = gen(z)

        pred_disc_false = disc(fake.detach())
        loss_disc_fake = F.binary_cross_entropy_with_logits(pred_disc_false, ones)

        loss_disc_true.backward()
        loss_disc_fake.backward()

        loss_disc = (loss_disc_true + loss_disc_fake) /2
        optim_disc.step()

        optim_gen.zero_grad()

        pred_disc_false = disc(fake)
        loss_gen = F.binary_cross_entropy_with_logits(pred_disc_false, zeros)

        loss_gen.backward()
        optim_gen.step()

        epoch_gen_loss+=loss_gen.item()
        epoch_disc_loss+=loss_disc_fake.item() + loss_disc_true.item() 


    pbar.set_description(f"{epoch+1}/{EPOCHS}; gen loss: {epoch_gen_loss/len(trainloader)}, disc loss: {epoch_disc_loss/len(trainloader)}")

checkpoint = {"gen":gen.state_dict(), "disc":disc.state_dict()}
torch.save(checkpoint, 'checkpoints/madgan.pkl')

gen.eval()
disc.eval()

valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

val_scores = []
weight = 0.95

for x_cont, _, _, _ in valloader:
    x_cont = x_cont[:, :, :-2]
    x_cont = x_cont.to(DEVICE)
    batch_latent = get_best_latent(gen, x_cont.clone(), latent_dim=LATENT_DIM)
    with torch.no_grad():
        generated = gen(batch_latent)
        disc_score = torch.sigmoid(disc(generated))
    disc_score = torch.squeeze(disc_score)
    res_loss = (generated - x_cont).abs().sum(dim=(1, 2))
    score = weight*disc_score + (1-weight)*res_loss
    val_scores.append(-score)

val_scores = torch.cat(val_scores)

test_scores = []
weight = 0.95

for x_cont, _, _, _ in testloader:
    x_cont = x_cont[:, :, :-2]
    x_cont = x_cont.to(DEVICE)
    batch_latent = get_best_latent(gen, x_cont.clone(), latent_dim=LATENT_DIM)
    with torch.no_grad():
        generated = gen(batch_latent)
        disc_score = torch.sigmoid(disc(generated))
    disc_score = torch.squeeze(disc_score)
    res_loss = (generated - x_cont).abs().sum(dim=(1, 2))
    score = weight*disc_score + (1-weight)*res_loss
    test_scores.append(-score)

test_scores = torch.cat(test_scores)

test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores, dim=1)) / (len(val_scores) + 1)
test_p_values = test_p_values.cpu()

np.save('pvalues/madgan.npy', test_p_values.numpy())

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

plt.savefig('figures/madgan/prederrors.png')