import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(root="data", window_size=10, batch_size=32):

    normal = pd.read_csv(os.path.join(root, 'WADI_14days_new.csv'), sep=",", skip_blank_lines=True)

    normal = normal.drop(normal.columns[[0,1,2,50,51,86,87]],axis=1)
    normal = normal.astype(float)

    scaler = StandardScaler()
    x_normal = normal.values
    x_normal_scaled = scaler.fit_transform(x_normal)
    normal = pd.DataFrame(x_normal_scaled)

    normal = normal.fillna(0)

    attack = pd.read_csv(os.path.join(root,'WADI_attackdataLABLE.csv'), skiprows=1)

    attack=attack.drop(attack.columns[[0,1,2,50,51,86,87]],axis=1)

    labels = attack[attack.columns[-1]].values
    attack = attack.drop(attack.columns[-1], axis=1)

    attack = attack.astype(float)

    x_attack = attack.values
    x_attack_scaled = scaler.transform(x_attack)
    attack = pd.DataFrame(x_attack_scaled)

    windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
    windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]

    trainset = TensorDataset(torch.from_numpy(windows_normal).to(torch.float32), torch.ones(len(windows_normal)))
    testset = TensorDataset(torch.from_numpy(windows_attack).to(torch.float32), torch.tensor(labels[window_size:]))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader =  DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

