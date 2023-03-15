"""
Script to prepare data for figure 2 and generate plots in the style that are shown in paper
"""

from figure2.figure2_prepare_data import prepare_data
from figure2.figure2_plot_data import plot_hist_figure
from reproducible_ephys_functions import get_histology_insertions, get_insertions
from one.api import ONE


def run_figure2(one, recompute=False, supplementary=False, freeze='release_2022_11'):
    # download all repeated site insertions
    _ = get_insertions(level=0, one=one, freeze=freeze)
    insertions = get_histology_insertions(one=one, freeze=freeze)
    concat_df_chns, concat_df_traj = prepare_data(insertions, one=one, recompute=recompute)
    plot_hist_figure(raw_histology=False)
    if supplementary:
        print('Raw histology data to make figure 2 supplementary is not yet released')


if __name__ == '__main__':
    one = ONE()
    run_figure2(one)
