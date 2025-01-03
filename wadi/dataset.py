import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(root="data", window_size=10, batch_size=32):
    try:
        normal = pd.read_csv(os.path.join(root, 'WADI_14days_new.csv'), sep=",", skip_blank_lines=True)
        normal = normal.drop(normal.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)
        normal = normal.astype(np.float32)

        scaler = StandardScaler()
        x_normal = normal.values
        x_normal_scaled = scaler.fit_transform(x_normal)
        normal = pd.DataFrame(x_normal_scaled).fillna(0)

        attack = pd.read_csv(os.path.join(root, 'WADI_attackdataLABLE.csv'), skiprows=1)
        attack = attack.drop(attack.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)

        labels = attack.iloc[:, -1].values.astype(np.float32)
        attack = attack.drop(attack.columns[-1], axis=1).astype(np.float32)

        x_attack = attack.values
        x_attack_scaled = scaler.transform(x_attack)
        attack = pd.DataFrame(x_attack_scaled).fillna(0)

        windows_normal = normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size)[:, None]]
        windows_attack = attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size)[:, None]]
        labels = labels[window_size:]

        trainset = TensorDataset(torch.from_numpy(windows_normal).to(torch.float32), torch.ones(len(windows_normal)))
        testset = TensorDataset(torch.from_numpy(windows_attack).to(torch.float32), torch.tensor(labels))

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

        return trainloader, testloader

    except Exception as e:
        print(f"Error: {e}")
        raise

