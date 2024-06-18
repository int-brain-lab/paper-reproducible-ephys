import pandas as pd
from pathlib import Path

tables_dir = Path(__file__).parent

clusters = pd.read_parquet(tables_dir.joinpath("clusters_ibl_jaccard_matched.pqt"))
channels = pd.read_parquet(tables_dir.joinpath("channels_ibl_jaccard_matched.pqt"))
channels_rms = pd.read_parquet(tables_dir.joinpath("channels_bwm_noiserms.pqt"))
channels_w = pd.read_parquet(tables_dir.joinpath("channels_bwm_WM.pqt"))

ins_use = clusters.index.get_level_values(0).unique()
ins_chan = channels.index.get_level_values(0).unique()

assert len(set(ins_use).difference(ins_chan)) == 0

channels["noise_rms"] = 0.0
channels["wm"] = 0.0
for p in ins_chan:
    new_table = channels.loc[p]
    wm_table = channels_w.loc[p] # contains the noise rms column

    assert len(new_table) == len(wm_table) == 384

    channels.loc[p,["noise_rms", "wm"]] = wm_table[["noise_rms", "wm"]].values

channels.to_parquet(tables_dir.joinpath("channels_bwm.pqt"))