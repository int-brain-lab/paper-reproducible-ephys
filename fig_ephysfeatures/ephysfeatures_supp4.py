#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:34:00 2022
By: Guido Meijer
"""

from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from reproducible_ephys_functions import save_figure_path, figure_style, LAB_MAP
from fig_ephysfeatures.ephysfeatures_plot_functions import panel_example
import numpy as np

REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap_p90', 'spike_amp_mean']
LABELS = ['Neuron yield\n(neurons/channel)', 'Firing rate\n(spikes/s)', 'LFP power (dB)', 'AP band RMS (\u03bcV)', 'Spike amp. (\u03bcV)']
N_REC_PER_REGION = 3


def plot_figure_supp4():
    figure_style()

    width = 7
    height = 6
    fig = plt.figure(figsize=(width, height), dpi=300)
    gs = fig.add_gridspec(len(METRICS) + 1, len(REGIONS),
                          height_ratios=np.r_[np.ones(len(METRICS)) * 1, 0.3])
    gs.update(wspace=0.3)

    for i, metric in enumerate(METRICS):
        ax_metric = []
        for j, region in enumerate(REGIONS):
            if j == 0:
                ylabel = LABELS[i]
            else:
                ylabel = ''

            ax = fig.add_subplot(gs[i, j])
            ax_metric.append(ax)
            panel_example(ax, n_rec_per_region=N_REC_PER_REGION, ylabel=ylabel,
                          example_metric=metric, example_region=region,
                          despine=False, freeze='freeze_2024_03')
            ax.set_xticklabels([])
            if i == 0:
                if region == 'PPC':
                    ax.set_title('VISa/am')
                else:
                    ax.set_title(region)
            if i == len(METRICS) - 1:
                ax.set_xlabel('Labs')

        max_y = np.max([ax.get_ylim()[1] for ax in ax_metric])
        min_y = np.min([ax.get_ylim()[0] for ax in ax_metric])
        for ax in ax_metric:
            ax.set_ylim(min_y, max_y)

    ax = fig.add_subplot(gs[len(METRICS), :])
    ax.set_axis_off()
    lab_number_map, institution_map, institution_colors = LAB_MAP()
    inst = list(set(list(institution_map.values())))
    inst.sort()
    for i, l in enumerate(inst):
        if l == 'UCL (H)':
            continue
        if i == 0:
            text = ax.text(0.2, -0.2, l, color=institution_colors[l], fontsize=8, transform=ax.transAxes)
        else:
            text = ax.annotate(
                '  ' + l, xycoords=text, xy=(1, 0), verticalalignment="bottom",
                color=institution_colors[l], fontsize=8)  # custom properties

    sns.despine()
    adjust = 0.3
    fig.subplots_adjust(top=1 - adjust / height, bottom=adjust / height, left=(adjust + 0.3) / width,
                        right=1 - adjust / width)
    plt.savefig(join(save_figure_path(figure='fig_ephysfeatures'), 'figure3_supp4.pdf'))


if __name__ == '__main__':
    plot_figure_supp4()
