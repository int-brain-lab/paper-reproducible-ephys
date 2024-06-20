from fig_data_quality import (plot_neuron_yield, plot_qualitative_ratings,
                              plot_supplemental_neuron_metrics,
                              save_neuron_yield_anova,
                              save_qualitative_ratings_anova)
from reproducible_ephys_functions import save_data_path, save_figure_path

def save_data_quality_anovas():
    data_path = save_data_path(figure="fig_data_quality")
    print(f'Saving CSVs to {data_path}')

    save_neuron_yield_anova()
    save_qualitative_ratings_anova()


def run_fig_data_quality():

    fig_path = save_figure_path(figure="fig_data_quality")
    print(f'Saving figures to {fig_path}')

    plot_neuron_yield()
    plot_supplemental_neuron_metrics()
    plot_qualitative_ratings()

    save_data_quality_anovas()



if __name__ == "__main__":

    run_fig_data_quality()