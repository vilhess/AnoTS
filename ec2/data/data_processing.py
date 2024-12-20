import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../../Datasets/TimeSeries/ec2_request_latency_system_failure.csv")

with open("data/config.json", "r") as file:
    config = json.load(file)

anomaly_periods = [
    (pd.to_datetime(period["start"]), pd.to_datetime(period["end"]))
    for period in config["anomaly_periods"]
]

data['timestamp']=pd.to_datetime(data['timestamp'])

data["anomalie"]=0
for start, end in anomaly_periods:
    data.loc[(data["timestamp"] >= start) & (data["timestamp"] <= end), "anomalie"] = 1

val_date = pd.to_datetime(config["val_date"])
test_date = pd.to_datetime(config["test_date"])

data['hour_min'] = data['timestamp'].dt.hour + data['timestamp'].dt.minute / 60

data['day_of_week'] = data['timestamp'].dt.dayofweek

cont_vars = ["value", "hour_min"]
cat_vars = ["day_of_week"]

label_encoders = [LabelEncoder() for _  in cat_vars]
for col, enc in zip(cat_vars, label_encoders):
    data[col]=enc.fit_transform(data[col])

trainset = data.loc[data["timestamp"] <= pd.to_datetime(val_date)].copy()
valset = data.loc[(data["timestamp"] > pd.to_datetime(val_date)) & (data["timestamp"] <= pd.to_datetime(test_date))].copy()
testset = data.loc[data["timestamp"] > pd.to_datetime(test_date)].copy()

scaler = StandardScaler().fit(trainset[cont_vars])

train_data_normed = trainset.copy()
val_data_normed = valset.copy()
test_data_normed = testset.copy()

train_data_normed[cont_vars] = scaler.transform(train_data_normed[cont_vars])
val_data_normed[cont_vars] = scaler.transform(val_data_normed[cont_vars])
test_data_normed[cont_vars] = scaler.transform(test_data_normed[cont_vars])

train_data_normed.to_csv("data/trainset.csv", index=False)
val_data_normed.to_csv('data/valset.csv', index=False)
test_data_normed.to_csv("data/testset.csv", index=False)