"""
Script to prepare data for figure 6 and generate plots in the style that are shown in paper
"""

from fig_spatial.fig_spatial_prepare_data import prepare_data, default_params
from fig_spatial.fig_spatial_plot_data import plot_supp_figure
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_fig_spatial(one, recompute=False):

    insertions = get_insertions(level=0, recompute=recompute, one=one, freeze='freeze_2024_03')
    prepare_data(insertions, one=one, **default_params, recompute=recompute)
    plot_supp_figure()


if __name__ == '__main__':
    one = ONE()
    run_fig_spatial(one)
