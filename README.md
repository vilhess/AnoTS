# AnoTS

In this repository, we implement several anomaly detection algorithms under an unsupervised framework. Currently, our work focuses on two datasets for the moment:  

- **NYC Taxi Demand Dataset** (located in the `nyctaxi` folder)  
- **EC2 Request Dataset** (located in the `ec2` folder)  

Both datasets are sourced from the Numenta Anomaly Benchmark (NAB) and can be found [here](https://github.com/numenta/NAB/).

## Models

The models we consider are:
- **LSTM predictive model** 
- **Transformer predictive model** 
- [**Deep One-Class Classification**](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf)
- [**MAD-GAN**](https://arxiv.org/abs/1901.04997)
- **LSTM Auto-Encoders** 
- [**DROCC**](https://arxiv.org/abs/2002.12718)
- [**TranAD**](https://arxiv.org/abs/2201.07284)

## To-Do:

- Test models on others datasets

- Implement other models: 
    - [ImDiffusion](https://www.vldb.org/pvldb/vol17/p359-zhang.pdf)