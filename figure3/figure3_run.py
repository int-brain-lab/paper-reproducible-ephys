"""
Script to prepare data for figure 3 and generate plots in the style that are shown in paper
"""
import logging

from figure3.figure3_prepare_data import prepare_data
from figure3.figure3_plot_data import plot_main_figure
import reproducible_ephys_functions
from one.api import ONE

logger = logging.getLogger('paper_reproducible_ephys')


def run_figure3(one, recompute=False, supplementary=False, freeze='release_2022_11'):
    insertions = reproducible_ephys_functions.get_insertions(level=0, one=one, freeze=freeze)
    if recompute:
        reproducible_ephys_functions.compute_metrics(insertions, one=one)
    prepare_data(insertions, one=one, recompute=recompute)
    plot_main_figure(one=one, freeze=freeze)
    if supplementary:
        from figure3.figure3_supp1 import plot_figure_supp1
        # from figure3.figure3_supp2 import plot_figure_supp2
        plot_figure_supp1(freeze=freeze)
        # plot_figure_supp2()


if __name__ == '__main__':
    # Use this for the open access data
    #one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)

    # Use this for the latest data
    # one = ONE(base_url='https://openalyx.internationalbrainlab.org')
    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    # Create figure
    run_figure3(one=one, recompute=True)
