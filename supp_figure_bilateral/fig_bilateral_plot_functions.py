from reproducible_ephys_functions import filter_recordings, labs, BRAIN_REGIONS, query, get_insertions
import pandas as pd
import numpy as np
from figure3.figure3_functions import get_brain_boundaries, plot_probe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ibllib.atlas.regions import BrainRegions
from supp_figure_bilateral.load_data import load_dataframe
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from matplotlib.sankey import Sankey
from permutation_test import permut_test, permut_dist
from statsmodels.stats.multitest import multipletests

br = BrainRegions()

lab_number_map, institution_map, lab_colors = labs()


def panel_probe_lfp(fig, ax, boundary_align='DG-TH', ylim=[-2000, 2000],
                    normalize=False, clim=[-190, -150]):

    df_chns = load_dataframe(df_name='chns')
    df_chns['institute'] = df_chns['lab'].map(institution_map)
    df_chns['lab_number'] = df_chns['lab'].map(lab_number_map)
    df_filt = df_chns.copy()
    df_filt = df_filt.sort_values(by=['institute', 'subject', 'probe']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset=['subject', 'pid']).reset_index()
    df_filt['recording'] = [i[-1] for i in df_filt['probe']]
    df_filt['recording'] = df_filt['recording'].map({'0': 'L', '1': 'R'})

    for iR, data in df_filt.iterrows():
        df = df_chns[df_chns['pid'] == data['pid']]

        la = {}
        la['id'] = df['region_id'].values
        z = df['z'].values * 1e6

        boundaries, colours, regions = get_brain_boundaries(la, z)
        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract

        # Plot
        im = plot_probe(df['lfp'].values, z, ax[iR], clim=clim, normalize=normalize,
                        cmap='viridis')

        ax[iR].set_title(data['recording'], color=lab_colors[data['institute']])
        if iR == 0:
            ax[iR].set(yticks=np.arange(ylim[0], ylim[1] + 1, 500),
                       yticklabels=np.arange(ylim[0], ylim[1] + 1, 500) / 1000,
                       xticks=[])
            ax[iR].tick_params(axis='y')
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set_ylabel('Depth relative to DG-Thalamus (mm)')
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)
    ax[-1].set_axis_off()

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="90%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    if normalize:
        cbar.ax.set_yticklabels(['10th\nperc.', '90th\nperc'])
    else:
        cbar.ax.set_yticklabels([f'{clim[0]}', f'{clim[1]}'])
    cbar.set_label('Power spectral density (dB)', rotation=270, labelpad=-5)


def panel_probe_neurons(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000]):

    df_chns = load_dataframe(df_name='chns')
    df_clust = load_dataframe(df_name='clust')

    df_chns['institute'] = df_chns['lab'].map(institution_map)
    df_chns['lab_number'] = df_chns['lab'].map(lab_number_map)
    df_filt = df_chns.copy()
    df_filt = df_filt.sort_values(by=['institute', 'subject', 'probe']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset=['subject', 'pid']).reset_index()
    df_filt['recording'] = [i[-1] for i in df_filt['probe']]
    df_filt['recording'] = df_filt['recording'].map({'0': 'L', '1': 'R'})

    for iR, data in df_filt.iterrows():
        df_ch = df_chns[df_chns['pid'] == data['pid']]
        df_clu = df_clust[df_clust['pid'] == data['pid']]

        la = {}
        la['id'] = df_ch['region_id'].values
        z = df_ch['z'].values * 1e6
        boundaries, colours, regions = get_brain_boundaries(la, z)
        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract
        else:
            z_subtract = 0

        levels = [0, 30]
        im = ax[iR].scatter(np.log10(df_clu['amps'] * 1e6), df_clu['depths_aligned'] - z_subtract, c=df_clu['fr'], s=1,
                            cmap='hot', vmin=levels[0], vmax=levels[1], zorder=2)
        ax[iR].images.append(im)
        ax[iR].set_xlim(1.3, 3)

        # First for all regions
        region_info = br.get(df_ch['region_id'].values)
        boundaries = np.where(np.diff(region_info.id) != 0)[0]
        boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
        regions = z[np.c_[boundaries[0:-1], boundaries[1:]]]
        region_colours = region_info.rgb[boundaries[1:]]

        width = ax[iR].get_xlim()[1]
        for reg, col in zip(regions, region_colours):
            height = np.abs(reg[1] - reg[0])
            color = col / 255
            ax[iR].bar(x=width / 2, height=height, width=width, color='grey', bottom=reg[0],
                       edgecolor='w', alpha=0.5, zorder=0)

        # Now for rep site
        region_info = br.get(df_ch['region_id_rep'].values)
        boundaries = np.where(np.diff(region_info.id) != 0)[0]
        boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
        regions = z[np.c_[boundaries[0:-1], boundaries[1:]]]
        region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]
        region_labels[region_labels[:, 1] == 'VISa', 1] = 'PPC'
        region_colours = region_info.rgb[boundaries[1:]]
        reg_idx = np.where(np.isin(region_labels[:, 1], BRAIN_REGIONS))[0]
#
        for i, (reg, col, lab) in enumerate(zip(regions, region_colours, region_labels)):
            height = np.abs(reg[1] - reg[0])
            if np.isin(i, reg_idx):
                alpha = 1
                color = col / 255
            else:
                alpha = 0
                color = 'grey'
            ax[iR].bar(x=width / 2, height=height, width=width, color=color, bottom=reg[0],
                       edgecolor='k', alpha=alpha, zorder=1)

        ax[iR].set_title(data['recording'], color=lab_colors[data['institute']])

        if iR == 0:
            ax[iR].set(yticks=np.arange(ylim[0], ylim[1] + 1, 500),
                       yticklabels=np.arange(ylim[0], ylim[1] + 1, 500) / 1000,
                       xticks=[])
            ax[iR].tick_params(axis='y')
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set_ylabel('Depth relative to DG-Thalamus (mm)')
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)

    # Add brain regions
    width = ax[-1].get_xlim()[1]
    ax[-1].set(ylim=ylim)
    ax[-1].bar(x=width / 2, height=750, width=width, color=np.array([0, 159, 172]) / 255,
               bottom=1250, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=500, width=width, color=np.array([126, 208, 75]) / 255,
               bottom=650, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=500, width=width, color=np.array([126, 208, 75]) / 255,
               bottom=50, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=900, width=width, color=np.array([255, 144, 159]) / 255,
               bottom=-950, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=950, width=width, color=np.array([255, 144, 159]) / 255,
               bottom=-2000, edgecolor='k', linewidth=0)
    ax[-1].text(width / 2 + 0.1, 1600, 'PPC', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2 + 0.1, 900, 'CA1', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2 + 0.1, 300, 'DG', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2 + 0.1, -500, 'LP', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2 + 0.1, -1500, 'PO', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].set_axis_off()

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    cbar.ax.set_yticklabels([f'{levels[0]}', f'{levels[1]}'])
    cbar.set_label('Firing rate (spks/s)', rotation=270, labelpad=-2)


def panel_example(ax, n_rec_per_lab=0, n_rec_per_region=3,
                  example_region='CA1', example_metric='lfp_power', ylabel='LFP power in CA1 (db)',
                  ylim=None, yticks=None, despine=True):

    df_ins = load_dataframe(df_name='ins')
    df_chns = load_dataframe(df_name='chns')



    df_ins['lab_number'] = df_ins['lab'].map(lab_number_map)
    df_ins['yield_per_channel'] = df_ins['neuron_yield'] / df_ins['n_channels']
    df_ins.loc[df_ins['lfp_power'] < -100000, 'lfp_power'] = np.nan
    data = df_ins.copy()

    data_example = pd.DataFrame(data={
        'institute': data.loc[data['region'] == example_region, 'institute'],
        'lab_number': data.loc[data['region'] == example_region, 'lab_number'],
        example_metric: data.loc[data['region'] == example_region, example_metric].values})

    data_example = data_example.sort_values('lab_number')
    cmap = []
    for i, inst in enumerate(data_example['institute'].unique()):
        cmap.append(lab_colors[inst])

    sns.stripplot(data=data_example, x='institute', y=example_metric, palette=cmap, s=3, ax=ax)
    ax_lines = sns.pointplot(x='institute', y=example_metric, data=data_example,
                             ci=0, join=False, estimator=np.mean, color='k',
                             markers="_", scale=1, ax=ax)

    #plt.setp(ax_lines.collections, zorder=100, label="")
    ax.plot(np.arange(data_example['institute'].unique().shape[0]),
             [data_example[example_metric].mean()] * data_example['institute'].unique().shape[0],
             color='r', lw=1)
    ax.set(ylabel=ylabel, xlabel='', xlim=[-.5, len(data['institute'].unique()) + .5])
    if ylim is not None:
        ax.set(ylim=ylim)
    if yticks is not None:
        ax.set(yticks=yticks)
    ax.set_xticklabels(data_example['institute'].unique(), rotation=90, ha='right')

    if despine:
        sns.despine(trim=True)

