#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:26:44 2022
By: Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import figrid as fg
from reproducible_ephys_functions import save_figure_path, figure_style
from figure3.figure3_plot_functions import panel_decoding
from figure3.figure3_load_data import load_dataframe

REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
save_path = save_figure_path(figure='figure3')

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

# Set up figure
figure_style()
fig = plt.figure(figsize=(8, 5), dpi=200)  # full width figure is 7 inches
ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.05, 0.28], yspan=[0.05, 0.4]),
      'panel_B': fg.place_axes_on_grid(fig, xspan=[0.32, 0.62], yspan=[0.05, 0.4]),
      'panel_C': fg.place_axes_on_grid(fig, xspan=[0.7, 1], yspan=[0.05, 0.4]),
      'panel_D': fg.place_axes_on_grid(fig, xspan=[0.05, 0.28], yspan=[0.65, 1]),
      'panel_E': fg.place_axes_on_grid(fig, xspan=[0.32, 0.62], yspan=[0.65, 1]),
      'panel_F': fg.place_axes_on_grid(fig, xspan=[0.7, 1], yspan=[0.65, 1])}


# Add subplot labels
labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold',
           'ha': 'left', 'va': 'bottom'},
          {'label_text':'b', 'xpos':0.32, 'ypos':0, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'},
          {'label_text':'c', 'xpos':0.65, 'ypos':0, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'},
          {'label_text':'d', 'xpos':0, 'ypos':0.6, 'fontsize':10, 'weight': 'bold',
           'ha': 'left', 'va': 'bottom'},
          {'label_text':'e', 'xpos':0.32, 'ypos':0.6, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'},
          {'label_text':'f', 'xpos':0.65, 'ypos':0.6, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'}]
fg.add_labels(fig, labels)

p_qc = panel_decoding(ax['panel_A'], qc=True)
ax['panel_A'].set(title='QC recordings')

ax['panel_B'].imshow(decode_qc_df.loc[0:4, METRICS].to_numpy())
ax['panel_B'].set(xticks=np.arange(4.1), yticks=np.arange(4.1), xticklabels=LABELS, yticklabels=REGIONS,
        title='Decoding importance')
ax['panel_B'].set_xticklabels(ax['panel_B'].get_xticklabels(), rotation=30, ha='right')

plt_hndl = ax['panel_C'].imshow(conf_mat_qc.to_numpy())
ax['panel_C'].set(xticks=np.arange(conf_mat_qc.shape[0]), yticks=np.arange(conf_mat_qc.shape[0]),
        xticklabels=conf_mat_qc.columns.values, yticklabels=conf_mat_qc.columns.values,
        title='Confusion matrix')
cbar = plt.colorbar(plt_hndl, ax=ax['panel_C'], shrink=0.5)
ax['panel_C'].set_xticklabels(ax['panel_C'].get_xticklabels(), rotation=90, ha='right')
#cbar.set_label('', rotation=270)
#cbar.ax.get_yaxis().labelpad = 8

p_all = panel_decoding(ax['panel_D'], qc=False)
ax['panel_D'].set(title='All recordings')

ax['panel_E'].imshow(decode_noqc_df.loc[0:4, METRICS].to_numpy())
ax['panel_E'].set(xticks=np.arange(4.1), yticks=np.arange(4.1), xticklabels=LABELS, yticklabels=REGIONS,
        title='Decoding importance')
ax['panel_E'].set_xticklabels(ax['panel_E'].get_xticklabels(), rotation=30, ha='right')

plt_hndl = ax['panel_F'].imshow(conf_mat_noqc.to_numpy())
ax['panel_F'].set(xticks=np.arange(conf_mat_noqc.shape[0]), yticks=np.arange(conf_mat_noqc.shape[0]),
                  xticklabels=conf_mat_noqc.columns.values, yticklabels=conf_mat_noqc.columns.values,
                  title='Confusion matrix')
cbar = plt.colorbar(plt_hndl, ax=ax['panel_F'], shrink=0.5)
ax['panel_F'].set_xticklabels(ax['panel_F'].get_xticklabels(), rotation=90, ha='right')
#cbar.set_label('', rotation=270)
#cbar.ax.get_yaxis().labelpad = 8


plt.tight_layout()
plt.savefig(save_path.joinpath('figure3_supp2.pdf'))
plt.savefig(save_path.joinpath('figure3_supp2.png'))

