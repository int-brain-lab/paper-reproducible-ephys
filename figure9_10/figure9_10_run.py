"""
Script to prepare data for figures 9 and 10 and generate plots in the style that are shown in paper
"""
import sys 
sys.path.append('..')
from figure9_10.figure9_10_prepare_data import prepare_data
from figure9_10.figure9_10_load_data import download_data
from figure9_10.figure9_train import train as train_fig9
from figure9_10.figure10_train import train as train_fig10
from figure9_10.figure9_plot_data import plot_figures as plot_figures9
from figure9_10.figure10_plot_data import plot_figures as plot_figures10

from one.api import ONE


def run_figure9_10(one, do_training=False):

    if not do_training:
        download_data()
    else:
        prepare_data(one)
        train_fig9()
        train_fig10()

    plot_figures9()
    plot_figures10()


if __name__ == '__main__':
    one = ONE()
    run_figure9_10(one)
