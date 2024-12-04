from fig_taskmodulation.fig_taskmodulation_prepare_data import (prepare_data, default_params,
                                                                compute_permutation_test, compute_power_analysis,
                                                                compute_permutation_test_modulated)
from fig_taskmodulation.fig_taskmodulation_plot_data import plot_main_figure, plot_power_analysis, plot_supp_figure
from reproducible_ephys_functions import get_insertions
from one.api import ONE


def run_fig_taskmodulation(one, recompute=False):

    insertions = get_insertions(level=0, one=one, freeze='freeze_2024_03')
    prepare_data(insertions, one=one, **default_params, recompute=recompute)
    compute_permutation_test(n_permut=40000, n_cores=8)
    compute_permutation_test_modulated(n_permut=40000, n_cores=8)
    compute_power_analysis(n_permut=40000, n_cores=8)

    plot_main_figure()
    plot_power_analysis()
    plot_supp_figure()


if __name__ == '__main__':
    one = ONE()
    run_fig_taskmodulation(one)