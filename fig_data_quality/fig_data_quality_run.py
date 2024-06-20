from reproducible_ephys_functions import save_figure_path
from fig_data_quality import plot_neuron_yield, plot_supplemental_neuron_metrics, plot_qualitative_ratings

def plot_data_quality():

    fig_path = save_figure_path(figure="fig_data_quality")
    print(f'Saving figures to {fig_path}')

    plot_neuron_yield()
    plot_supplemental_neuron_metrics()
    plot_qualitative_ratings()

if __name__ == "__main__":

    plot_data_quality()