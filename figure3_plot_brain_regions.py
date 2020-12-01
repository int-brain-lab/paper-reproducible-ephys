#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:48:00 2020

@author: guido
"""

from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from reproducible_ephys_functions import data_path, labs
from reproducible_ephys_paths import FIG_PATH

# Settings
REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']

# Load in data
metrics = pd.read_csv(join(data_path(), 'figure3_brain_regions.csv'))

# Get lab info
lab_number_map, institution_map, lab_colors = labs()

# Reformat data
metrics.loc[metrics['n_channels'] == 0, 'neuron_yield'] = np.nan
metrics.loc[metrics['lfp_power_low'] < -100000, 'lfp_power_low'] = np.nan
metrics.loc[metrics['lfp_power_high'] < -100000, 'lfp_power_high'] = np.nan
metrics['institution'] = metrics.lab.map(institution_map)
for i, region in enumerate(REGIONS):
    metrics.loc[metrics['region'] == region, 'region_number'] = i
metrics = metrics.sort_values(by=['institution', 'eid', 'region_number'], ignore_index=True).reset_index(drop=True)
n_rec = np.concatenate([np.arange(i) + 1 for i in (metrics.groupby('institution').size()
                                                   / len(REGIONS))]).astype(int)
metrics['yield_per_channel'] = metrics['neuron_yield'] / metrics['n_channels']
for i, eid in enumerate(metrics['eid'].unique()):
    metrics.loc[metrics['eid'] == eid, 'recording_number'] = i

# %% Plot neuron yield
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='yield_per_channel').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0,
            cbar_kws={'label': 'Neurons per channel'}, annot=True, linewidths=.5, fmt='.2f')
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_neuron_yield'))

# %% Plot firing rate
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='firing_rate').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0,
            cbar_kws={'label': 'Firing rate (Hz)'}, annot=True, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_firing_rate'))

# %% Plot spike amplitude
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='spike_amp_mean').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.0f',
            cbar_kws={'label': 'Spike amplitude (mV)'}, annot=True, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_spike_amplitude'))

# %% Plot peak-to-trough ratio
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='pt_ratio').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.2f',
            cbar_kws={'label': 'Peak-to-trough ratio'}, annot=True, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_pt_ratio'))

# %% Plot repolarization slope
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='rp_slope').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.0f',
            cbar_kws={'label': 'Repolarization slope'}, annot=True, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_rp_slope'))

# %% Plot LFP power
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='lfp_power_low').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=metrics['lfp_power_low'].min(),
            cbar_kws={'label': 'LFP power (dB)'}, annot=True, linewidths=.5, fmt='.0f')
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_lfp_power_low'))

sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='lfp_power_high').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=metrics['lfp_power_high'].min(),
            cbar_kws={'label': 'LFP power (dB)'}, annot=True, linewidths=.5, fmt='.0f')
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_lfp_power_high'))

#%%  Plot rms AP
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 4), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='rms_ap').sort_values('region_number')
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.0f',
            cbar_kws={'label': 'AP band (rms)'}, annot=True, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=n_rec)
ax1.set_yticklabels(REGIONS, va='center')
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, -0.8, inst, ha='center', color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_rms_ap'))


