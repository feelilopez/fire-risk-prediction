# Wildfire Risk Prediction in Spain

This project aims to support wildfire risk prediction in Spain by providing a more comprehensive evaluation of the [IberFire dataset](https://arxiv.org/abs/2505.00837) for baseline and deep learning models. We will focus on predicting the probability of fire occurrence for the next day based on the previous week's data. The project will involve data sampling, cleaning, model training, and evaluation, with an emphasis on handling class imbalance and ensuring robust performance metrics.


## Environment setup

Use a dedicated virtual environment and install dependencies from
`requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use Jupyter, select the `.venv` interpreter as the notebook kernel.


## Data sampling (`2-sampling.ipynb`)

This notebook implements data sampling from the IberFire dataset to handle imbalance and data size before training our models. It creates a training set with a 1:3 ratio of fire to no-fire samples, and generates three types of negative samples for each fire sample. It also creates unbalanced validation and test sets with a 1:10 ratio of fire to no-fire samples.

We create datasets for two model architectures: a baseline (which we call XGBoost for simplicity) and an LSTM. The baseline dataset includes only the features from the sample day (day_offset = 0), while the LSTM dataset includes a 7-day window of data for each sample (day_offset = 0..6).

The sampled datasets are saved in Parquet format for later use in model training. See `data/baseline` folder for the baseline datasets and `data/lstm` folder for the LSTM datasets.
