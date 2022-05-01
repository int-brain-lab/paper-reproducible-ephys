"""
Script to prepare data for figure 6 and generate plots in the style that are shown in paper
"""

from figure7.figure7_prepare_data import prepare_data, default_params
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_figure7(one, recompute=False, supplementary=False):

    insertions = get_insertions(level=2, one=one, freeze='biorxiv_2022_05')
    prepare_data(insertions, one=one, **default_params, recompute=recompute)


if __name__ == '__main__':
    one = ONE()
    run_figure7(one)
