import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../../Datasets/TimeSeries/nyc_taxi.csv")

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

data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month
data['hour_min'] = data['timestamp'].dt.hour + data['timestamp'].dt.minute / 60

data['day_of_week'] = data['timestamp'].dt.dayofweek
data['holiday'] = 0

data.loc[(data['day'] == 25) & (data['month'] == 12),'holiday'] = 1  # Christmas
data.loc[(data['day'] == 1) & (data['month'] == 1),'holiday'] = 1  # New Year's Day

holidays = data.loc[data['holiday'] == 1, 'timestamp'].dt.date.unique()

for i, hd in enumerate(holidays):
    data['hol_' + str(i)] = data['timestamp'].dt.date - hd

for i in range(data.shape[0]):
    if np.abs(data.loc[data.index[i], 'hol_0']) <= np.abs(data.loc[data.index[i], 'hol_1']):
        data.loc[data.index[i], 'gap_holiday'] = data.loc[data.index[i], 'hol_0']
    else:
        data.loc[data.index[i], 'gap_holiday'] = data.loc[data.index[i], 'hol_1']

data['gap_holiday'] = data['gap_holiday'].dt.days
data.drop(['hol_0', 'hol_1'], axis=1, inplace=True)

data['t'] = (data['timestamp'].astype(np.int64)/1e11).astype(np.int64)

cont_vars = ["value", "hour_min", "gap_holiday", "t"]
cat_vars = ["day", "month", "day_of_week", "holiday"]

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