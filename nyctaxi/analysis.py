import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from dataset import get_datasets

st.set_page_config(layout="wide")

st.title('New York City Taxi Demand Dataset - Anomaly Detections')

st.subheader('Anomalies: NYC Marathon / Thanksgiving / Christmas / New Year / Snow Storm')

WINDOW=10
BATCH_SIZE=128

if 'dataset' not in st.session_state:
    trainset, valset, testset, dataset = get_datasets(window=WINDOW, lbl_as_feat=False)
    st.session_state['dataset'] = dataset
    st.session_state['trainset'] = trainset
    st.session_state['valset'] = valset
    st.session_state['testset'] = testset
else:
    dataset = st.session_state["dataset"]
    trainset = st.session_state['trainset']
    valset = st.session_state['valset']
    testset = st.session_state['testset']

model = st.sidebar.selectbox(label="Model: ", options=["LSTM", "DOC", "AELSTM", "TRANSAM", "DROCC", "TranAD", "MAD-GAN", "PatchTST"])
model = model.lower().replace('-', '')

threshold = st.sidebar.slider(label="threshold: ", min_value=0.001, max_value=0.05, step=0.001, value=0.01, format="%.3f" )
revin=None
if model in (["lstm", "transam", "patchtst"]):
    revin = st.sidebar.checkbox("Use revin?")
ext = "_rev" if revin else ""

train_indices = dataset.train_indices
val_indices = dataset.val_indices
test_indices = dataset.test_indices

test_anomalies = dataset.test_anomalies

results = {}
models = os.listdir('pvalues')

for model_pval in models:
    model_str = model_pval.split(".")[0]
    results[model_str] = []

    p_values = np.load(os.path.join('pvalues', model_pval))
    p_values = np.squeeze(p_values)

    df = pd.DataFrame({"p_values": p_values}, index=test_indices)
    df.index = pd.to_datetime(df.index)

    df['anomalie'] = 1-df.index.isin(test_anomalies).astype(int)

    auc = round(roc_auc_score(df["anomalie"], df["p_values"]), 3)
    results[model_str].append(auc)

    reject = p_values<threshold
    f1 = round(f1_score(df["anomalie"],reject), 3)
    prec = round(precision_score(df["anomalie"], reject), 3)
    recall = round(recall_score(df["anomalie"], reject), 3)

    results[model_str].append(f1)
    results[model_str].append(prec)
    results[model_str].append(recall)

results = pd.DataFrame(results)
results.index = ["AUC Score", "F1 Score", "Recall", "Precision"]

st.divider()
st.header("Table of results for all models considering the current threshold")
st.table(results)
st.divider()

traindf = dataset.dataframe[dataset.dataframe["timestamp"].isin(train_indices)]
valdf = dataset.dataframe[dataset.dataframe["timestamp"].isin(val_indices)]
testdf = dataset.dataframe[dataset.dataframe["timestamp"].isin(test_indices)]

traindf.index = pd.to_datetime(train_indices)
valdf.index = pd.to_datetime(val_indices)
testdf.index = pd.to_datetime(test_indices)

anomaly_periods = dataset.anomaly_periods

full_plot = Image.open("figures/ts.png")
full_plot_norm = Image.open("figures/normalize_ts.png")
valtest = Image.open("figures/valtest.png")
prederrors = Image.open(f"figures/{model}/prederrors{ext}.png")

st.image(full_plot)
st.image(full_plot_norm)
st.divider()

if model=="lstm":
    st.header('Bidirectional LSTM - Prediction Error')
elif model=="doc":
    st.header('Bidirectional LSTM - Deep One-Class Classifier')
elif model=="aelstm":
    st.header('AutoEncoder LSTM - Reconstruction Error')
elif model=="transam":
    st.header('Transformer TS - Prediction Error')
elif model=="drocc":
    st.header('LSTM-DROCC - Probability of being normal')
elif model=="mad-gan":
    st.header('MAD-GAN - Normality Score')
elif model=="tranad":
    st.header('TranAD - Normality Score')
elif model=="patchtst":
    st.header('PatchTST - Prediction error')

p_values = np.load(f"pvalues/{model}{ext}.npy")

st.image(valtest)
st.image(prederrors)

anomalies_indices = np.where(p_values <= threshold)[0]

fig = plt.figure(figsize=(15, 7))

plt.plot(testdf.index, testdf['value'], label="Test", c="blue")
plt.plot(testdf.index[[i for i in anomalies_indices]], testdf.iloc[[i for i in anomalies_indices]]['value'], 'ro', label='Anomalies detected')

for start, end in anomaly_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Period')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Time Series with Anomaly Periods - After Normalize', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.grid(True)

st.pyplot(fig)