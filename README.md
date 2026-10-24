# Multivariate Time Series Forecasting with Adaptive Non-Local Observables

Official implementation for the paper: **Multivariate Time Series Forecasting with Adaptive Non-Local Observables**.

We implement a data re-uploading variational quantum circuit (VQC) for multivariate time-series forecasting (MTSF). Specifically, fixed local observables are replaced with adaptive non-local observables (ANO) that jointly measure combinatorial subsets of $k$ qubits for better function approximation.

## Short Abstract
**TL;DR:** We apply adaptive non-local observables to multivariate time series forecasting.

Conventional quantum neural networks typically rely on fixed local measurements, which restrict their expressivity. We propose MTSF-ANO, a simple hybrid model for MTSF that addresses this limitation by integrating data re-uploading VQCs with ANO. On the four ETT datasets, MTSF-ANO ranks first or second in 17 of 20 settings, improving over the strongest baseline by up to 20\% on ETTh1, and matches or outperforms its fixed local observable counterpart across all settings. Supported by ablations on circuit design and ANO non-locality, these results suggest that ANO is a promising direction for quantum time series forecasting.

## Usage

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

## Repository Layout

```
models/          MTSF-ANO, its channel-independent variant, and RevIN
baselines/       Classical and quantum baselines, with upstream notices
data_provider/   ETT datasets and data loaders
exp/             Training and evaluation loop
scripts/         train.sh: one model on one dataset over several seeds
dataset/         ETT-small CSVs
```
