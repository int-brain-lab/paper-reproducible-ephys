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
QC = ['pass', 'all', 'artifacts', 'high_lfp', 'high_noise', 'low_trials', 'low_yield', 'missed_target']
TITLE = ['QC pass', 'All', 'Artifacts', 'High LFP', 'High noise', 'Low trials', 'Low yield', 'Missed target']
save_path = save_figure_path(figure='figure3')

# Set up figure
figure_style()
f, axs = plt.subplots(2, 4, figsize=(7, 3), dpi=400)
axs = np.concatenate(axs)

p_values = dict()
for i, this_qc in enumerate(QC):
    p_values[this_qc] = panel_decoding(axs[i], qc=this_qc)
    axs[i].set(title=TITLE[i])

plt.tight_layout()

