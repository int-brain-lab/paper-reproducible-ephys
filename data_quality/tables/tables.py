import pandas as pd
from pathlib import Path

tables_dir = Path(__file__).parent

dsets = ["allen", "steinmetz", "re", "bwm", "IBL_like_steinmetz", "IBL_like_allen", "re_147", "bwm_147", "re_147_recomputed"]

def _check_dset(dset):
    if dset not in dsets:
        raise ValueError(f"Must be one of: {dsets}")

def load_clusters(dset, filter_region=None):
    _check_dset(dset)
    path = tables_dir.joinpath(f"clusters_{dset}.pqt")
    pqt = pd.read_parquet(path)
    # firing rate cutoff
    pqt = pqt[pqt.firing_rate > 0.02]
    if filter_region:
        pqt = pqt[pqt.cosmos_acronym == filter_region]
    pqt.index = pqt.index.remove_unused_levels()
    pqt.index.set_names(["insertion", "unit_id"], inplace=True)
    return pqt.copy()

def load_channels(dset, filter_region=None):
    _check_dset(dset)
    path = tables_dir.joinpath(f"channels_{dset}.pqt")
    pqt = pd.read_parquet(path)
    if filter_region:
        pqt = pqt[pqt.cosmos_acronym == filter_region]
    pqt.index = pqt.index.remove_unused_levels()
    pqt.index.set_names(["insertion", "channel_id"], inplace=True)
    return pqt.copy()