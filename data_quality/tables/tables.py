import pandas as pd
from pathlib import Path

tables_dir = Path(__file__).parent

dsets = ["allen", "steinmetz", "re", "bwm"]

def _check_dset(dset):
    if dset not in dsets:
        raise ValueError(f"Must be one of: {dsets}")

def load_clusters(dset):
    _check_dset(dset)
    path = tables_dir.joinpath(f"clusters_{dset}.pqt")
    return pd.read_parquet(path)

def load_channels(dset):
    _check_dset(dset)
    path = tables_dir.joinpath(f"channels_{dset}.pqt")
    returjn pd.read_parquet(path)