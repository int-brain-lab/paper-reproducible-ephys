"""
Script to prepare data for figure 2 and generate plots in the style that are shown in paper
"""

from fig_hist.fig_hist_prepare_data import prepare_data
from fig_hist.fig_hist_plot_data import plot_hist_figure_new
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_fig_hist(one, recompute=False, supplementary=False, perform_permutation_test=False, freeze='freeze_2024_03'):
    # download all repeated site insertions
    insertions = get_insertions(level=-1, one=one, freeze=freeze)
    concat_df_chns, concat_df_traj = prepare_data(insertions, one=one, recompute=recompute)
    plot_hist_figure_new(perform_permutation_test=perform_permutation_test)
    if supplementary:
        print('Raw histology data to make figure histology supplementary is not yet released')


if __name__ == '__main__':
    one = ONE()
    run_fig_hist(one)
