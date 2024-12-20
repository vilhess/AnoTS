import torch
from torch.utils.data import Dataset
import pandas as pd
import json
    
class TSDataset(Dataset):
    def __init__(self, root="data", window=10, lbl_as_feat=True):
        super().__init__()

        # return cont_vars (+ value if lbl as feat), cat_vars, value, if anomaly ; batch of size window

        with open(f"{root}/config.json", "r") as file:
            config = json.load(file)

        self.anomaly_periods = [
            (pd.to_datetime(period["start"]), pd.to_datetime(period["end"]))
            for period in config["anomaly_periods"]
        ]

        self.cont_vars = ["value", "hour_min"]
        self.cat_vars = ["day_of_week"]

        self.window = window
        self.lbl_as_feat = lbl_as_feat


        trainset = pd.read_csv(f"{root}/trainset.csv")
        valset = pd.read_csv(f"{root}/valset.csv")
        testset = pd.read_csv(f"{root}/testset.csv")

        self.train_indices = trainset.loc[window:, "timestamp"]
        self.val_indices = valset["timestamp"]
        self.test_indices = testset['timestamp']

        self.dataset = pd.concat([trainset, valset, testset], ignore_index=True)

        self.dataframe = self.dataset.copy()
        self.dataset.drop("timestamp", axis=1, inplace=True)
        
        self.anomalies = self.dataset["anomalie"].copy()
        self.dataset.drop("anomalie", axis=1, inplace=True)

    def __len__(self):
        return len(self.dataset) - self.window

    def __getitem__(self, index):
        start_idx = index
        end_idx = index + self.window
        window_data_cont = self.dataset.loc[start_idx:end_idx+self.lbl_as_feat-1, self.cont_vars].values
        window_data_cat = self.dataset.loc[start_idx:end_idx+self.lbl_as_feat-1, self.cat_vars].values
        return torch.Tensor(window_data_cont), torch.Tensor(window_data_cat), torch.tensor(self.dataset.loc[end_idx]["value"], dtype=torch.float32), torch.tensor(self.anomalies.loc[end_idx])
    
def get_datasets(root="data", window=10, lbl_as_feat=True):
    ds = TSDataset(root=root, window=window, lbl_as_feat=lbl_as_feat)

    indices = range(len(ds))

    train_size = len(ds.train_indices)
    val_size = len(ds.val_indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    trainset = [ds[i] for i in train_indices]
    valset = [ds[i] for i in val_indices]
    testset = [ds[i] for i in test_indices]
    return trainset, valset, testset, ds