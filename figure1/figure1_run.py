"""
Script to prepare data for figure 1 and generate plots in the style that are shown in paper
"""
import logging

from figure1.figure1_prepare_data import prepare_data
from figure1.figure1_plot_data import plot_main_figure, plot_supp2_figure
from ibllib.atlas import AllenAtlas
from one.api import ONE

logger = logging.getLogger('paper_reproducible_ephys')


def run_figure1(one, supplementary=True):

    ba = AllenAtlas()
    prepare_data(one, ba=ba)
    plot_main_figure(one, ba=ba)
    if supplementary:
        plot_supp2_figure(one, ba=ba)


if __name__ == '__main__':
    one = ONE()
    run_figure1(one)
