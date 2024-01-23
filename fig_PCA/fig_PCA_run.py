"""
Script to prepare data for figure 6 and generate plots in the style that are shown in paper
"""

from fig_PCA.fig_PCA_prepare_data import prepare_data, default_params
from fig_PCA.fig_PCA_plot_data import all_panels
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_fig_PCA(one, recompute=False, freeze='release_2022_11'):

    insertions = get_insertions(level=0, one=one, freeze=freeze)
    prepare_data(insertions, one=one, **default_params, recompute=recompute)
    all_panels(freeze=freeze)


if __name__ == '__main__':
    one = ONE()
    run_fig_PCA(one)
