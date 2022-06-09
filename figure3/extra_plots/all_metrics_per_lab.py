#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:34:00 2022
By: Guido Meijer
"""

import matplotlib.pyplot as plt
from figure3.figure3_plot_functions import panel_example

REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
N_REC_PER_REGION = 2

f, axs = plt.subplots(len(REGIONS), len(METRICS), figsize=(12, 8), dpi=300)

for i, region in enumerate(REGIONS):
    for j, metric in enumerate(METRICS):
        panel_example(axs[i, j], n_rec_per_region=N_REC_PER_REGION, ylabel=LABELS[j],
                      example_metric=metric, example_region=region,
                      despine=False)
        axs[i, j].set(title=region)
plt.tight_layout()
