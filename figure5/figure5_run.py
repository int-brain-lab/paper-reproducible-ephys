"""
Script to prepare data for figure 5 and generate plots in the style that are shown in paper
"""

from figure5.figure5_prepare_data import prepare_data, default_params
from figure5.figure5_plot_data import plot_main_figure
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_figure5(one, recompute=False):

    insertions = get_insertions(level=2, one=one, freeze='biorxiv_2022_05')
    prepare_data(insertions, one=one, **default_params, recompute=recompute)
    plot_main_figure()


if __name__ == '__main__':
    one = ONE()
    run_figure5(one)
