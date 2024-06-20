"""
Script to prepare data for figure 3 and generate plots in the style that are shown in paper
"""
import logging

from fig_ephysfeatures.ephysfeatures_prepare_data import prepare_data, run_decoding
from fig_ephysfeatures.ephysfeatures_plot_data import plot_main_figure
import reproducible_ephys_functions
from one.api import ONE

logger = logging.getLogger('paper_reproducible_ephys')


def run_fig_ephysfeatures(one, recompute=False, supplementary=True, bilateral=False, freeze='freeze_2024_03'):
    insertions = reproducible_ephys_functions.get_insertions(level=0, one=one, freeze=freeze)
    prepare_data(insertions, one=one, recompute=recompute)
    run_decoding(n_shuffle=500, qc='pass', recompute=recompute)
    run_decoding(n_shuffle=500, qc='all', recompute=recompute)

    plot_main_figure(one=one, freeze=freeze)
    if supplementary:
        if bilateral:
            import fig_ephysfeatures.supp_figure_bilateral.prepare_data as supp_prepare_data
            insertions = reproducible_ephys_functions.query(min_regions=0, n_trials=0, behavior=False,
                                                            exclude_critical=True, one=one,
                                                            as_dataframe=False, bilateral=True)
            reproducible_ephys_functions.compute_metrics(insertions, one=one, bilateral=True)
            supp_prepare_data.prepare_neural_data(insertions, recompute=recompute, one=one)
            _ = supp_prepare_data.prepare_data(insertions, recompute=recompute, one=one)


        from fig_ephysfeatures.ephysfeatures_supp1 import plot_figure_supp1
        from fig_ephysfeatures.ephysfeatures_supp2 import plot_figure_supp2
        from fig_ephysfeatures.ephysfeatures_supp4 import plot_figure_supp4
        import fig_ephysfeatures.supp_figure_bilateral.plot_data as supp_plot_data
        plot_figure_supp1()
        plot_figure_supp2(freeze=freeze)
        supp_plot_data.plot_main_figure()
        plot_figure_supp4()


if __name__ == '__main__':

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    # Create figure
    run_fig_ephysfeatures(one=one, recompute=False, supplementary=False, freeze='freeze_2024_03')
