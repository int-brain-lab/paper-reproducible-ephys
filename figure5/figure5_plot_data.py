import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from reproducible_ephys_functions import BRAIN_REGIONS, labs, figure_style
from reproducible_ephys_paths import DATA_PATH
import numpy as np

_, _, lab_colors = labs()

tests = {'trial': 'Trial',
         'start_to_move': 'Pre move (TW)',
         'post_stim': 'Post stim',
         'pre_move': 'Pre move',
         'pre_move_lr': 'Move LvR',
         'post_move': 'Post move',
         'post_reward': 'Post reward',
         'avg_ff_post_move': 'FanoFactor'}

# load dataframe
save_path = Path(DATA_PATH).joinpath('figure5')
df = pd.read_csv(save_path.joinpath('figure5_figure6_dataframe'))

# Group data frame by region
df_region = df.groupby('region')
restriction = 'la'


# FIGURE 5c and supplementary figures
for test in tests.keys():
    for i, br in enumerate(BRAIN_REGIONS):
        plt.subplot(len(BRAIN_REGIONS), 1, i + 1)
        df_br = df_region.get_group(br)

        df_inst = df_br.groupby(['subject', 'institute'], as_index=False)
        vals = df_inst[test].mean().sort_values('institute')
        colors = [lab_colors[col] for col in vals['institute'].values]
        plt.bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
        plt.ylim(bottom=0, top=1)
        plt.ylabel(br)
        plt.xticks([])
        if i == 4:
            plt.xlabel('Mice')
    plt.suptitle(tests[test], size=22)
    plt.savefig(save_path.joinpath(f"{test}_{restriction}"))
    plt.close()

min_recordings_per_lab = 4
for i, br in enumerate(BRAIN_REGIONS):

    df_br = df_region.get_group('CA1')
    df_filt = df_br.groupby('institute').filter(lambda s: s['subject'].nunique() >= min_recordings_per_lab)

    # etc etc

