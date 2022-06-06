"""
Script to prepare data for figure 6 and generate plots in the style that are shown in paper
"""

from figure6.figure6_prepare_data import prepare_data, default_params
from figure6.figure6_plot_data import all_panels
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_figure6(one, recompute=False, new_metrics=True):

    insertions = get_insertions(level=2, one=one, freeze=None, new_metrics=new_metrics)
    prepare_data(insertions, one=one, **default_params, recompute=recompute, new_metrics=new_metrics)
    all_panels()


if __name__ == '__main__':
    one = ONE()
    run_figure6(one)
