# Anomaly Detection for the EC2 Dataset

## Overview
This project focuses on detecting anomalies in the **CPU usage data** from a server in Amazon's East Coast datacenter. The dataset captures the system's performance leading up to a complete system failure caused by a documented failure of AWS API servers.

The dataset can be accessed [here](https://www.kaggle.com/datasets/boltzmannbrain/nab).

## Project Structure
- **`models/`**  
  Contains the implementation of machine learning models used for anomaly detection.

- **`trainers/`**  
  Includes the anomaly detection algorithms used to train and compute the p-values.

- **`data/`**  
  Contains preprocessing scripts and datasets.  
  - Data preprocessing involves normalization using a training set.  

- **`analysis.py`**  
  A Streamlit-based web interface for interactive analysis.  
  Run it using the command:  
  ```bash
  streamlit run analysis.py
  ```

You can experiment with various parameters, such as:

- Selecting different models.
- Adjusting the threshold for anomaly detection.