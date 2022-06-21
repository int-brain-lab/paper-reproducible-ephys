#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:26:44 2022
By: Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
from figure3.figure3_plot_functions import panel_decoding
from figure3.figure3_load_data import load_dataframe

REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']

# Load in data
decode_qc_df = load_dataframe(df_name='decode')
shuffle_qc_df = load_dataframe(df_name='decode_shuf')
decode_noqc_df = load_dataframe(df_name='decode_no_qc')
shuffle_noqc_df = load_dataframe(df_name='decode_shuf_no_qc')
conf_mat_qc = load_dataframe(df_name='conf_mat')
conf_mat_noqc = load_dataframe(df_name='conf_mat_no_qc')

# Restructure
conf_mat_qc = conf_mat_qc.drop(['Unnamed: 0'], axis=1)
conf_mat_noqc = conf_mat_noqc.drop(['Unnamed: 0'], axis=1)

# Plot
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 5), dpi=400)

p_qc = panel_decoding(ax1, qc=True)
ax1.set(title='QC recordings')

ax2.imshow(decode_qc_df.loc[0:4, METRICS].to_numpy())
ax2.set(xticks=np.arange(4.1), yticks=np.arange(4.1), xticklabels=LABELS, yticklabels=REGIONS,
        title='Decoding importance')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')

plt_hndl = ax3.imshow(conf_mat_qc.to_numpy())
ax3.set(xticks=np.arange(conf_mat_qc.shape[0]), yticks=np.arange(conf_mat_qc.shape[0]),
        xticklabels=conf_mat_qc.columns.values, yticklabels=conf_mat_qc.columns.values,
        title='Confusion matrix')
cbar = plt.colorbar(plt_hndl, ax=ax3, shrink=0.5)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, ha='right')
#cbar.set_label('', rotation=270)
#cbar.ax.get_yaxis().labelpad = 8

p_all = panel_decoding(ax4, qc=False)
ax4.set(title='All recordings')

ax5.imshow(decode_noqc_df.loc[0:4, METRICS].to_numpy())
ax5.set(xticks=np.arange(4.1), yticks=np.arange(4.1), xticklabels=LABELS, yticklabels=REGIONS,
        title='Decoding importance')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right')

plt_hndl = ax6.imshow(conf_mat_noqc.to_numpy())
ax6.set(xticks=np.arange(conf_mat_noqc.shape[0]), yticks=np.arange(conf_mat_noqc.shape[0]),
        xticklabels=conf_mat_noqc.columns.values, yticklabels=conf_mat_noqc.columns.values,
        title='Confusion matrix')
cbar = plt.colorbar(plt_hndl, ax=ax6, shrink=0.5)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90, ha='right')
#cbar.set_label('', rotation=270)
#cbar.ax.get_yaxis().labelpad = 8

plt.tight_layout()

