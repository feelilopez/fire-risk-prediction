# Fire Risk Prediction

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

This notebook implements the data sampling strategy described in the report. It creates a training set with a 1:3 ratio of fire to no-fire samples, and generates three types of negative samples for each fire sample. It also creates unbalanced validation and test sets with a 1:10 ratio of fire to no-fire samples.

We create datasets for two model architectures: a baseline (which we call XGBoost for simplicity) and an LSTM. The baseline dataset includes only the features from the sample day (day_offset = 0), while the LSTM dataset includes a 7-day window of data for each sample (day_offset = 0..6).

The sampled datasets are saved in Parquet format for later use in model training. See `data/baseline` folder for the baseline splitted datasets and `data/lstm` folder for the LSTM datasets.