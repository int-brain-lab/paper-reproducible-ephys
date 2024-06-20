from fig_data_quality import (plot_neuron_yield, plot_qualitative_ratings,
                              plot_supplemental_neuron_metrics,
                              save_neuron_yield_anova,
                              save_qualitative_ratings_anova)
from reproducible_ephys_functions import save_data_path, save_figure_path


def plot_data_quality():

    fig_path = save_figure_path(figure="fig_data_quality")
    print(f'Saving figures to {fig_path}')

    plot_neuron_yield()
    plot_supplemental_neuron_metrics()
    plot_qualitative_ratings()

def save_data_quality_anovas():
    data_path = save_data_path(figure="fig_data_quality")
    print(f'Saving CSVs to {data_path}')

    save_neuron_yield_anova()
    save_qualitative_ratings_anova()

if __name__ == "__main__":

    plot_data_quality()
    save_data_quality_anovas()