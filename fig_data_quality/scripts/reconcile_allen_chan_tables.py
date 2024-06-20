import pandas as pd
from pathlib import Path

tables_dir = Path(__file__).parent

clusters_allen = pd.read_parquet(tables_dir.joinpath("clusters_allen.pqt"))
channels_allen = pd.read_parquet(tables_dir.joinpath("channels_allen.pqt"))
channels_allen_rms = pd.read_parquet(tables_dir.joinpath("channels_allen_noiserms.pqt"))
channels_allen_w = pd.read_parquet(tables_dir.joinpath("channels_allen_WM.pqt"))

ins_use = clusters_allen.index.get_level_values(0).unique()
channels_new = channels_allen[channels_allen.index.get_level_values(0).isin(ins_use)]

ins_chan = channels_new.index.get_level_values(0).unique()
assert len(set(ins_use).difference(ins_chan)) == 0

channels_new["noise_rms"] = 0.0
channels_new["wm"] = 0.0
for p in ins_chan:
    new_table = channels_new.loc[p]
    wm_table = channels_allen_w.loc[p] # contains the noise rms column

    assert len(new_table) == len(wm_table) == 384

    channels_new.loc[p,["noise_rms", "wm"]] = wm_table[["noise_rms", "wm"]].values

channels_new.to_parquet(tables_dir.joinpath("channels_allen.pqt"))