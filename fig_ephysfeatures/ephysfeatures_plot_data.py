#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from fig_ephysfeatures.ephysfeatures_plot_functions import (
    panel_probe_lfp, panel_probe_neurons, panel_permutation, panel_decoding,
    panel_example)
import matplotlib.pyplot as plt
import pickle
from reproducible_ephys_functions import figure_style, filter_recordings, save_figure_path, labs
from one.api import ONE
import numpy as np


def plot_main_figure(freeze=None, one=None):

    one = one or ONE()

    # Settings
    MIN_REC_PER_LAB = 3  # for plotting of probe plots
    MIN_REC_PER_REGION = 3  # for permutation testing
    MIN_REGIONS = 2  # min regions hit to include recording
    BOUNDARY = 'DG-TH'
    REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
    METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power',
               'rms_ap', 'spike_amp_mean']
    LABELS = ['Neuron yield', 'Firing rate', 'LFP power',
              'AP band RMS', 'Spike amp.']
    N_PERMUT = 50000  # Amount of shuffles for permutation testing
    BH_CORRECTION = False  # Correction for multiple comparisons
    #N_PERMUT = 50  # Amount of shuffles for permutation testing
    DPI = 150  # if the figure is too big on your screen, lower this number
    np.random.seed(42)  # fix the random seed for reproducible permutatation results

    # Get filtered dataframe 
    lab_number_map, institution_map, lab_colors = labs()
    df_filt = filter_recordings(min_rec_lab=MIN_REC_PER_LAB, min_regions=MIN_REGIONS)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['institute', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('institute', group_keys=False).size()
    df_filt['recording'] = np.mod(np.concatenate([np.arange(i) for i in rec_per_lab.values]), 10)
    n_columns = len(df_filt['subject'].unique())

    # Set up figure
    figure_style()
    fig = plt.figure(figsize=(7, 9), dpi=DPI)  # full width figure is 7 inches
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0, 0.18]),
         'panel_B': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.25, 0.5],
                                          dim=[1, n_columns + 1], wspace=0.3),
         'panel_C': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.55, 0.8],
                                          dim=[1, n_columns + 1], wspace=0.3),
         'panel_D': fg.place_axes_on_grid(fig, xspan=[0.2, 0.35], yspan=[0.85, 1]),
         'panel_E': fg.place_axes_on_grid(fig, xspan=[0.66, 0.88], yspan=[0.85, 1])}
         #'panel_F': fg.place_axes_on_grid(fig, xspan=[0.78, 1], yspan=[0.85, 1])}

    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.2, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'c', 'xpos':0, 'ypos':0.55, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'d', 'xpos':0.1, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'e', 'xpos':0.57, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    #{'label_text':'g', 'xpos':0.7, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
    #'ha': 'right', 'va': 'bottom'}]
    fg.add_labels(fig, labels)

    # Call functions to plot panels
    ax['panel_A'].axis('off')
    pids_b = panel_probe_lfp(fig, ax['panel_B'], df_filt, boundary_align=BOUNDARY, freeze=freeze)
    
    pids_c = panel_probe_neurons(fig, ax['panel_C'], df_filt, boundary_align=BOUNDARY, freeze=freeze)
    
    
    p_permut, pids_d = panel_permutation(ax['panel_D'], METRICS, REGIONS, LABELS,
                                         n_permut=N_PERMUT,
                                         n_rec_per_lab=MIN_REC_PER_LAB,
                                         n_rec_per_region=MIN_REC_PER_REGION,
                                         bh_correction=BH_CORRECTION,
                                         freeze=freeze)
    
    p_decoding = panel_decoding(ax['panel_E'], qc='pass', bh_correction=True)
    
    
    """
    ax['panel_E'].set(title='QC pass')
    _ = panel_decoding(ax['panel_F'], qc='all')
    ax['panel_F'].set(title='All recordings')
    panel_example(ax['panel_F'], freeze=freeze)
    """
    
    # Save figure
    save_path = save_figure_path(figure='fig_ephysfeatures')
    print(f'Saving figures to {save_path}')
    plt.savefig(save_path.joinpath('figure3.png'))
    plt.savefig(save_path.joinpath('figure3.pdf'))
    
    # Save dict with pids
    dict_pids = dict()
    dict_pids['fig3'] = dict()
    dict_pids['fig3']['b'] = pids_b
    dict_pids['fig3']['c'] = pids_c
    dict_pids['fig3']['d'] = pids_d
    with open(save_path.joinpath('dict_pids.pkl'), 'wb') as fp:
        pickle.dump(dict_pids, fp)
    
    return p_decoding, p_permut


if __name__ == '__main__':
    one = ONE()
    p_decoding, p_permut = plot_main_figure(freeze='freeze_2024_03', one=one)
