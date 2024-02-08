# **Efficiency Meets Robustness: Enhancing Time Series Representations through Noise-Resilient Training and an Efficient Encoder**.

## Requirements

The `requirements.txt` file are attached for list of packages required.
* Python 3.9.16
* torch==2.0.0
* scikit_learn==0.24.2
* pywavelets==1.4.1
* pandas
* scipy
* statsmodels
* matplotlib
* Bottleneck

The dependencies can be installed with this single-line command:
```bash
pip install -r requirements.txt
```

## Datasets
The datasets are all publicly available online, put into `data/` folder in the following way:
```bash
cd ..
mkdir data/
cd data
```
* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018): After downloading and unzip-ing the compressed file, rename the folder to `UCR`.
* [30 UEA datasets](http://www.timeseriesclassification.com): After downloading and unzip-ing the compressed file, rename the folder to `UEA`.
* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset): Download 3 files `ETTh1.csv`, `ETTh2.csv` and `ETTm1.csv`.
* [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014): 
 After downloading and unzip-ing the compressed file, run preprocessing file at `CoInception/preprocessing/preprocess_electricity.py` and placed at `../data/electricity.csv`.
* [Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70): 
First register for using the dataset, then downloading and unzip-ing the compressed file, run preprocessing file at `CoInception/preprocessing/preprocess_yahoo.py` and placed at `../data/yahoo.pkl`.
* [KPI dataset](http://test-10056879.file.myqcloud.com/10056879/test/20180524_78431960010324/KPI%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%86%B3%E8%B5%9B%E6%95%B0%E6%8D%AE%E9%9B%86.zip): 
After downloading and unzip-ing the compressed file, run preprocessing file at `CoInception/preprocessing/preprocess_kpi.py` and placed at `../data/kpi.pkl`.


## Training and Evaluating

Run this one-line command for both training and evaluation:

```train & evaluate
python train.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval --save_ckpt
```
Example:
```bash
python -u train.py Chinatown UCR --loader UCR --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
```

The detailed descriptions about the arguments are as following:
| Parameter name | Description|
| --- | --- |
| dataset_name (required) | The dataset name |
| run_name (required) | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| batch_size | The batch size (defaults to 8) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |
| save_ckpt | Whether to save checkpoint (default: False)

(For descriptions of more arguments, run `python train.py -h`.)

**Scripts:** The scripts for reproduction are provided in `scripts/` folder.

## Acknowledgement 
This codebase is partially inherited from these below repositories, we want to express our thank to the authors:
* [TS2Vec](https://github.com/yuezhihan/ts2vec): TS2Vec: Towards Universal Representation of Time Series (AAAI-22)
* [TNC](https://github.com/sanatonek/TNC_representation_learning): Unsupervised Representation Learning for TimeSeries with Temporal Neighborhood Coding (ICLR 2021)
* [T-Loss](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries): Unsupervised Scalable Representation Learning for Multivariate Time Series (NeurIPS 2019)
