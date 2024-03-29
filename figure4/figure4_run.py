"""
Script to prepare data for figure 4 and generate plots in the style that are shown in paper
"""

from figure4.figure4_prepare_data import prepare_data, default_params
from figure4.figure4_plot_data import plot_main_figure
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_figure4(one, recompute=False, new_metrics=True):

    insertions = get_insertions(level=2, one=one, freeze='biorxiv_2022_05', new_metrics=new_metrics)
    prepare_data(insertions, one=one, **default_params, recompute=recompute, new_metrics=new_metrics)
    plot_main_figure()


if __name__ == '__main__':
    one = ONE()
    run_figure4(one)
