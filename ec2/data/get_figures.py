import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import json

df = pd.read_csv("../../Datasets/TimeSeries/ec2_request_latency_system_failure.csv", index_col="timestamp", parse_dates=True)
with open("data/config.json", "r") as file:
    config = json.load(file)

anomaly_periods = [
    (pd.to_datetime(period["start"]), pd.to_datetime(period["end"]))
    for period in config["anomaly_periods"]
]

val_date = pd.to_datetime(config["val_date"])
test_date = pd.to_datetime(config["test_date"])

trainset = df.loc[:pd.to_datetime(val_date)]
valset = df.loc[pd.to_datetime(val_date):pd.to_datetime(test_date)]
testset = df.loc[pd.to_datetime(test_date):]

plt.figure(figsize=(20, 10))

trainset['value'].plot(label='Train', c="blue")
valset['value'].plot(label='Val', c="green")
testset['value'].plot(label='Test', c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Time Series with Anomaly Periods', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.grid(True)

plt.savefig('figures/ts.png')
plt.close()

scaler = StandardScaler()
trainset_tr = scaler.fit_transform(trainset)
valset_tr = scaler.transform(valset)
testset_tr = scaler.transform(testset)

plt.figure(figsize=(20, 10))

plt.plot(trainset.index, trainset_tr, label="Train", c="blue")
plt.plot(valset.index, valset_tr, label="Val", c="green")
plt.plot(testset.index, testset_tr, label="Test", c="red")

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Time Series with Anomaly Periods - After Normalize', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.grid(True)

plt.savefig('figures/normalize_ts.png')
plt.close()