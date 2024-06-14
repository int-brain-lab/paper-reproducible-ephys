#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:34:00 2022
By: Guido Meijer
"""

from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from reproducible_ephys_functions import save_figure_path
from figure3.figure3_plot_functions import panel_example

REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield\n(neurons/channel)', 'Firing rate\n(spks/s)', 'LFP power (dB)', 'AP band RMS', 'Spike amp. (uV)']
N_REC_PER_REGION = 2

f, axs = plt.subplots(len(REGIONS), len(METRICS), figsize=(12, 10), dpi=100)

for i, region in enumerate(REGIONS):
    for j, metric in enumerate(METRICS):
        panel_example(axs[i, j], n_rec_per_region=N_REC_PER_REGION, ylabel=LABELS[j],
                      example_metric=metric, example_region=region,
                      despine=False, freeze='freeze_2024_03')
        if region == 'PPC':
            axs[i, j].set(title='VISa/am')
        else:
            axs[i, j].set(title=region)
plt.tight_layout()
sns.despine()
plt.savefig(join(save_figure_path(), 'all_metrics_per_lab.pdf'))
