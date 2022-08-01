import matplotlib.pyplot as plt
import numpy as np
from supp_figure_bilateral.load_data import load_neural_data, load_dataframe
from reproducible_ephys_functions import filter_recordings


def plot_comparative_psth():
    df_chns = load_dataframe(df_name='chns')
    df_filt = filter_recordings(df_chns)
    df_filt = df_filt[df_filt['include'] == 1].reset_index()
    data = load_neural_data(event='move', norm='subtract', smoothing='sliding')
    return df_filt, data

if __name__ == '__main__':
    df_chns, data = plot_comparative_psth()
