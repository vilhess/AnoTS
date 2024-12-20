import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

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

model = st.sidebar.selectbox(label="Model: ", options=["LSTM", "DOC", "AELSTM", "TRANSAM", "DROCC", "TranAD", "MAD-GAN"])
model = model.lower().replace('-', '')

threshold = st.sidebar.slider(label="threshold: ", min_value=0.001, max_value=0.05, step=0.001, value=0.01, format="%.3f" )

train_indices = dataset.train_indices
val_indices = dataset.val_indices
test_indices = dataset.test_indices

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
prederrors = Image.open(f"figures/{model}/prederrors.png")

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

p_values = np.load(f"pvalues/{model}.npy")

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

anomalies_detected_timestamp = testdf.index[[i for i in anomalies_indices]]

periods_names = ['nyc marathon', 'thanksgiving', 'christmas', 'new year', 'snow storm']

anomaly_periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in anomaly_periods]

results = []
for name, (start, end) in zip(periods_names, anomaly_periods):
    anomalies_in_period = anomalies_detected_timestamp[
        (anomalies_detected_timestamp >= start) & (anomalies_detected_timestamp <= end)
    ]
    
    period_length = len(testdf.loc[start:end])
    anomalies_in_period_count = len(anomalies_in_period)
    
    percentage = (anomalies_in_period_count / period_length) * 100 if period_length > 0 else 0

    results.append({
        "Period Name": name,
        "Start Date": start.date(),
        "End Date": end.date(),
        "Anomalies Count": anomalies_in_period_count,
        "Period Length": period_length,
        "Anomalies (%)": round(percentage, 2)
    })

st.divider()
results_df = pd.DataFrame(results)
st.header(f'Table Results, threshold = {threshold*100}%')
st.table(results_df)
