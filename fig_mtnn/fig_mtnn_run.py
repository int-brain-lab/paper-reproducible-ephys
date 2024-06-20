"""
Script to prepare data for figures 9 and 10 and generate plots in the style that are shown in paper
"""
import sys 
sys.path.append('..')
from fig_mtnn.fig_mtnn_prepare_data import prepare_data
from fig_mtnn.fig_mtnn_load_data import download_data
from fig_mtnn.fig_mtnn_train import train_all, train_groups, train_labID_exp
from fig_mtnn.fig_mtnn_plot_data import plot_figures9, plot_figures10

from one.api import ONE


def run_fig_mtnn(one, do_training=False):

    if not do_training:
        download_data()
    else:
        prepare_data(one)
        train_all()
        train_groups()
        train_labID_exp()

    plot_figures9()
    plot_figures10()


if __name__ == '__main__':
    one = ONE()
    run_fig_mtnn(one)
