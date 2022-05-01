"""
Script to prepare data for figure 3 and generate plots in the style that are shown in paper
"""
import logging

from figure3.figure3_prepare_data import prepare_data
from figure3.figure3_plot_data import plot_main_figure
from figure3.figure3_supp1 import plot_figure_supp1
from figure3.figure3_supp2 import plot_figure_supp2
from reproducible_ephys_functions import get_insertions
from one.api import ONE

logger = logging.getLogger('paper_reproducible_ephys')


def run_figure3(one, recompute=False, supplementary=True):
    insertions = get_insertions(level=0, one=one, freeze='biorxiv_2022_05')
    prepare_data(insertions, one=one, recompute=recompute)
    plot_main_figure(one)
    if supplementary:
        plot_figure_supp1()
        plot_figure_supp2()


if __name__ == '__main__':
    one = ONE()
    run_figure3(one)
