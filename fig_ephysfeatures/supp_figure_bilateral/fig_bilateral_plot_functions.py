from reproducible_ephys_functions import filter_recordings, labs, BRAIN_REGIONS, query, get_insertions
import pandas as pd
import numpy as np
from fig_ephysfeatures.ephysfeatures_functions import get_brain_boundaries, plot_probe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from iblatlas.regions import BrainRegions
from scipy.stats import mannwhitneyu, iqr
from fig_ephysfeatures.supp_figure_bilateral.load_data import load_dataframe as load_bilateral_data
from fig_ephysfeatures.ephysfeatures_load_data import load_dataframe as load_all_data
import seaborn as sns

br = BrainRegions()

lab_number_map, institution_map, lab_colors = labs()


def panel_probe_lfp(fig, ax, boundary_align='DG-TH', ylim=[-2000, 2000],
                    normalize=False, clim=[-190, -150]):

    df_chns = load_bilateral_data(df_name='chns')
    df_chns['institute'] = df_chns['lab'].map(institution_map)
    df_chns['lab_number'] = df_chns['lab'].map(lab_number_map)
    df_chns = df_chns[~df_chns.institute.isin(['UCLA', 'UW'])]

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

    df_chns = load_bilateral_data(df_name='chns')
    df_clust = load_bilateral_data(df_name='clust')

    df_chns['institute'] = df_chns['lab'].map(institution_map)
    df_chns['lab_number'] = df_chns['lab'].map(lab_number_map)

    df_chns = df_chns[~df_chns.institute.isin(['UCLA', 'UW'])]

    df_filt = df_chns.copy()
    df_filt = df_filt.sort_values(by=['institute', 'subject', 'probe']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset=['subject', 'pid']).reset_index()
    df_filt['recording'] = [i[-1] for i in df_filt['probe']]
    df_filt['recording'] = df_filt['recording'].map({'0': 'L', '1': 'R'})

    df_bilateral_ins = load_bilateral_data(df_name='ins')

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
        im = ax[iR].scatter(np.random.uniform(low=0.25, high=0.75, size=df_clu.shape[0]),
                            df_clu['depths_aligned'] - z_subtract, c=df_clu['fr'], s=1,
                            cmap='hot', vmin=levels[0], vmax=levels[1], zorder=2)
        #ax[iR].add_images(im)
        ax[iR].set_xlim(0, 1)

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


def panel_boxplot(ax, example_region='CA1', example_metric='lfp_power',
                  ylabel='LFP power diff. in CA1 (db)',
                  ylim=None, yticks=None, despine=True):

    df_ins = load_bilateral_data(df_name='ins')
    df_slice = df_ins[(df_ins['region'] == example_region)]

    # Calculate within animal variability
    within_var = np.empty(np.unique(df_slice['subject']).shape[0])
    across_var = []
    for i, subject in enumerate(np.unique(df_slice['subject'])):
        within_var[i] = np.abs(df_slice[(df_slice['subject'] == subject)
                                        & (df_slice['probe'] == 'probe00')][example_metric].values[0]
                               - df_slice[(df_slice['subject'] == subject)
                                          & (df_slice['probe'] == 'probe01')][example_metric].values[0])

        for j, other_sub in enumerate(np.unique(df_slice.loc[df_slice['subject'] != subject, 'subject'])):
            # Probe 00
            across_var.append(np.abs(df_slice[(df_slice['subject'] == subject)
                                        & (df_slice['probe'] == 'probe00')][example_metric].values[0]
                               - df_slice[(df_slice['subject'] == other_sub)
                                          & (df_slice['probe'] == 'probe00')][example_metric].values[0]))

            # Probe 01
            across_var.append(np.abs(df_slice[(df_slice['subject'] == subject)
                                        & (df_slice['probe'] == 'probe01')][example_metric].values[0]
                               - df_slice[(df_slice['subject'] == other_sub)
                                          & (df_slice['probe'] == 'probe01')][example_metric].values[0]))
    across_var = np.array(across_var)
    within_var = within_var[~np.isnan(within_var)]
    across_var = across_var[~np.isnan(across_var)]

    # Plot
    ax.boxplot([within_var, across_var], medianprops=dict(color='k'), showfliers=False, widths=0.5)
    ax.set(xticks=[1, 2], xticklabels=['Within', 'Across'], ylabel=ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    if yticks is not None:
        ax.set(yticks=yticks)
    sns.despine(trim=True)


def panel_distribution(ax, example_region='CA1', example_metric='lfp_power',
                       ylabel='LFP power diff. in CA1 (db)', yticks=None):

    # Load in bilateral data
    df_bilateral_ins = load_bilateral_data(df_name='ins')
    df_bilateral_ins['institute'] = df_bilateral_ins['lab'].map(institution_map)
    df_bilateral_ins = df_bilateral_ins[~df_bilateral_ins.institute.isin(['UCLA', 'UW'])]
    df_bl_slice = df_bilateral_ins[(df_bilateral_ins['region'] == example_region)]
    print(df_bl_slice)

    # Load in all data
    df_all_ins = load_all_data(df_name='ins')
    df_filt = filter_recordings(df_all_ins, min_lab_region=0, min_rec_lab=0, min_neuron_region=2,
                                recompute=False)
    df_all_slice = df_filt[(df_filt['region'] == example_region)]

    # Calculate within animal variability
    within_var = np.empty(np.unique(df_bl_slice['subject']).shape[0])
    print(within_var)
    for i, subject in enumerate(np.unique(df_bl_slice['subject'])):
        print(subject)
        within_var[i] = np.abs(df_bl_slice[(df_bl_slice['subject'] == subject)
                                        & (df_bl_slice['probe'] == 'probe00')][example_metric].values[0]
                               - df_bl_slice[(df_bl_slice['subject'] == subject)
                                          & (df_bl_slice['probe'] == 'probe01')][example_metric].values[0])
    within_var = within_var[~np.isnan(within_var)]

    # Calculate across animal variability of entire dataset
    across_var = []
    for i, subject in enumerate(np.unique(df_all_slice['subject'])):
        across_var.append(np.abs(df_all_slice[df_all_slice['subject'] == subject][example_metric].values[0]
                          - df_all_slice[df_all_slice['subject'] != subject][example_metric].values))
    across_var = np.concatenate(across_var)
    across_var = across_var[~np.isnan(across_var)]

    # Plot
    ax.violinplot(across_var, showextrema=False)
    ax.plot(np.ones(within_var.shape[0]), within_var, '_', color='k', markersize=10)
    #ax.spines['bottom'].set_visible(False)
    ax.set(ylabel=ylabel, xticks=[1])
    if yticks is not None:
        ax.set(yticks=yticks)
    ax.tick_params(bottom=False, labelbottom=False)
    sns.despine(trim=True)

    #ax.spines['bottom'].set_visible(False)


def panel_summary(ax, regions=['PPC', 'CA1', 'DG']):

    # Load in bilateral data
    df_ins = load_bilateral_data(df_name='ins')
    df_ins['institute'] = df_ins['lab'].map(institution_map)
    df_ins = df_ins[~df_ins.institute.isin(['UCLA', 'UW'])]

    # Load in all data
    df_all_ins = load_all_data(df_name='ins')
    df_filt = filter_recordings(df_all_ins, min_lab_region=0, min_rec_lab=0, min_neuron_region=2,
                                recompute=False)
    df_filt['yield_per_channel'] = df_filt['neuron_yield'] / df_filt['n_channels']

    # Get difference within vs across variance
    diff_df = pd.DataFrame()
    for r, region in enumerate(regions):
        df_bl_slice = df_ins[df_ins['region'] == region]
        df_all_slice = df_filt[df_filt['region'] == region]
        for m, metric in enumerate(['median_firing_rate', 'spike_amp_median', 'rms_ap', 'lfp_power',
                                    'yield_per_channel']):


            # Calculate across animal variability
            across_var = []
            for i, subject in enumerate(np.unique(df_all_slice['subject'])):
                across_var.append(np.abs(df_all_slice[df_all_slice['subject'] == subject][metric].values[0]
                                  - df_all_slice[df_all_slice['subject'] != subject][metric].values))
            across_var = np.concatenate(across_var)
            across_var = across_var[~np.isnan(across_var)]

            # Calculate within animal variability
            within_var = np.empty(np.unique(df_bl_slice['subject']).shape[0])
            for i, subject in enumerate(np.unique(df_bl_slice['subject'])):
                within_var[i] = np.abs(df_bl_slice[(df_bl_slice['subject'] == subject)
                                                & (df_bl_slice['probe'] == 'probe00')][metric].values[0]
                                       - df_bl_slice[(df_bl_slice['subject'] == subject)
                                                  & (df_bl_slice['probe'] == 'probe01')][metric].values[0])
            within_var = within_var[~np.isnan(within_var)]
            diff = (iqr(within_var) - iqr(across_var)) / (iqr(within_var) + iqr(across_var))
            diff_df = pd.concat((diff_df, pd.DataFrame(index=[diff_df.shape[0] + 1], data={
                'diff': diff, 'region': region, 'metric': metric})))

    diff_table = diff_df.pivot(index='region', columns='metric', values='diff')
    results_plot = diff_table.reindex(columns=['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_median'])
    results_plot = diff_table.reindex(index=regions)

    axin = inset_axes(ax, width="5%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
    # cmap = sns.color_palette('viridis_r', n_colors=20)
    # cmap[0] = [1, 0, 0]
    sns.heatmap(results_plot, cmap='RdYlGn', square=True,
                cbar=True, cbar_ax=axin, vmin=-0.8, vmax=0.8,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.8, 0, 0.8])
    cbar.set_label('Ratio var. within / across', rotation=270, labelpad=8)
    labels = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
    ax.set(xlabel='', ylabel='', xticks=np.arange(len(labels)) + 0.5, yticks=np.arange(len(regions)) + 0.5)
    ax.set_yticklabels(regions, va='center', rotation=0)
    ax.set_xticklabels(labels, rotation=45, ha='right')


