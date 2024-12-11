from reproducible_ephys_functions import filter_recordings, LAB_MAP, BRAIN_REGIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fig_ephysfeatures.ephysfeatures_functions import get_brain_boundaries, plot_probe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from iblatlas.regions import BrainRegions
from fig_ephysfeatures.ephysfeatures_load_data import load_dataframe
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
from permutation_test import permut_test, distribution_dist_approx_max
from statsmodels.stats.multitest import multipletests
from iblutil.numerical import ismember

br = BrainRegions()
PRINT_INFO = False

lab_number_map, institution_map, lab_colors = LAB_MAP()


def panel_probe_lfp(fig, ax, df_filt, boundary_align='DG-TH', ylim=[-2000, 2000],
                    normalize=False, clim=[-190, -150], freeze=None):

    df_chns = load_dataframe(df_name='chns')


    df_lim = df_chns.drop_duplicates(subset='pid')

    a_in, b_in = ismember(df_lim.pid.values, df_filt.pid.values)
    df_filt.loc[b_in, 'avg_dist'] = df_lim.avg_dist.values[a_in]
    df_filt = df_filt.sort_values(by=['institute', 'avg_dist'], ascending=[True, True]).reset_index(drop=True)
    rec_per_lab = df_filt.groupby('institute', group_keys=False).size()
    df_filt['recording'] = np.mod(np.concatenate([np.arange(i) for i in rec_per_lab.values]), 10)


    if PRINT_INFO:
        print(f'Figure 3 b')
        print(f'N_inst: {df_filt.institute.nunique()}, N_sess: {df_filt.eid.nunique()}, '
              f'N_mice: {df_filt.subject.nunique()}, N_cells: NA')

    for iR, data in df_filt.iterrows():
        df = df_chns[df_chns['pid'] == data['pid']]
        if len(df) == 0:
            print(f'pid {data["pid"]} not found!')
            continue

        la = {}
        la['id'] = df['region_id'].values
        z = df['z'].values * 1e6

        boundaries, colours, regions = get_brain_boundaries(la, z)
        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract

        # Plot
        im = plot_probe(df['lfp_destriped'].values, z, ax[iR], clim=clim, normalize=normalize,
                        cmap='viridis')

        ax[iR].set_title(data['recording'] + 1, color=lab_colors[data['institute']], size=6.5)
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

        # Add squigly line if probe plot is cut off
        if np.min(z) < np.min(ylim):
            ax[iR].text(ax[iR].get_xlim()[1] / 2, ylim[0] - 180, '~', fontsize=10, ha='center')

    ax[-1].set_axis_off()

    # Add lab names
    plt.figtext(0.12, 0.705, 'Berkeley', va="center", ha="center", size=7, color=lab_colors['Berkeley'])
    plt.figtext(0.215, 0.705, 'CCU', va="center", ha="center", size=7, color=lab_colors['CCU'])
    plt.figtext(0.300, 0.705, 'CSHL (C)', va="center", ha="center", size=7, color=lab_colors['CSHL (C)'])
    plt.figtext(0.345, 0.705, '(Z)', va="center", ha="center", size=7, color=lab_colors['CSHL (Z)'])
    plt.figtext(0.40, 0.705, 'NYU', va="center", ha="center", size=7, color=lab_colors['NYU'])
    plt.figtext(0.465, 0.705, 'Princeton', va="center", ha="center", size=7, color=lab_colors['Princeton'])
    plt.figtext(0.53, 0.705, 'SWC', va="center", ha="center", size=7, color=lab_colors['SWC'])
    plt.figtext(0.635, 0.705, 'UCL', va="center", ha="center", size=7, color=lab_colors['UCL'])
    #plt.figtext(0.805, 0.715, '(H)', va="center", ha="center", size=7, color=lab_colors['UCL (H)'])
    plt.figtext(0.78, 0.705, 'UCLA', va="center", ha="center", size=7, color=lab_colors['UCLA'])
    plt.figtext(0.89, 0.705, 'UW', va="center", ha="center", size=7, color=lab_colors['UW'])

    # Old positions Add lab names
    # plt.figtext(0.22, 0.715, 'Berkeley', va="center", ha="center", size=7, color=lab_colors['Berkeley'])
    # plt.figtext(0.305, 0.715, 'Champalimaud', va="center", ha="center", size=7, color=lab_colors['CCU'])
    # plt.figtext(0.385, 0.715, 'CSHL (C)', va="center", ha="center", size=7, color=lab_colors['CSHL (C)'])
    # plt.figtext(0.425, 0.715, '(Z)', va="center", ha="center", size=7, color=lab_colors['CSHL (Z)'])
    # plt.figtext(0.46, 0.715, 'NYU', va="center", ha="center", size=7, color=lab_colors['NYU'])
    # plt.figtext(0.52, 0.715, 'Princeton', va="center", ha="center", size=7, color=lab_colors['Princeton'])
    # plt.figtext(0.57, 0.715, 'SWC', va="center", ha="center", size=7, color=lab_colors['SWC'])
    # plt.figtext(0.645, 0.715, 'UCL', va="center", ha="center", size=7, color=lab_colors['UCL'])
    # #plt.figtext(0.805, 0.715, '(H)', va="center", ha="center", size=7, color=lab_colors['UCL (H)'])
    # plt.figtext(0.77, 0.715, 'UCLA', va="center", ha="center", size=7, color=lab_colors['UCLA'])
    # plt.figtext(0.86, 0.715, 'UW', va="center", ha="center", size=7, color=lab_colors['UW'])

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="90%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    if normalize:
        cbar.ax.set_yticklabels(['10th\nperc.', '90th\nperc'])
    else:
        cbar.ax.set_yticklabels([f'{clim[0]}', f'{clim[1]}'])
    cbar.set_label('Power spectral density (dB)', rotation=270, labelpad=-5)

    # Return the list of pids used in this figure
    return np.unique(df_filt['pid'])


def panel_probe_neurons(fig, ax, df_filt, boundary_align='DG-TH', ylim=[-2000, 2000], freeze=None):

    df_chns = load_dataframe(df_name='chns')
    df_clust = load_dataframe(df_name='clust')

    df_lim = df_chns.drop_duplicates(subset='pid')

    a_in, b_in = ismember(df_lim.pid.values, df_filt.pid.values)
    df_filt.loc[b_in, 'avg_dist'] = df_lim.avg_dist.values[a_in]
    df_filt = df_filt.sort_values(by=['institute', 'avg_dist'], ascending=[True, True]).reset_index(drop=True)
    rec_per_lab = df_filt.groupby('institute', group_keys=False).size()
    df_filt['recording'] = np.mod(np.concatenate([np.arange(i) for i in rec_per_lab.values]), 10)

    if PRINT_INFO:
        print(f'Figure 3 b')
        print(f'N_inst: {df_filt.institute.nunique()}, N_sess: {df_filt.eid.nunique()}, '
              f'N_mice: {df_filt.subject.nunique()}, N_cells: {len(df_filt)}')

    for iR, data in df_filt.iterrows():

        df_ch = df_chns[df_chns['pid'] == data['pid']]
        df_clu = df_clust[df_clust['pid'] == data['pid']]

        if len(df_ch) == 0:
            continue

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
        #ax[iR].add_image(im)
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

        ax[iR].set_title(data['recording'] + 1, color=lab_colors[data['institute']], size=6.5)

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

        # Add squigly line if probe plot is cut off
        if np.min(z) < np.min(ylim):
            ax[iR].text(ax[iR].get_xlim()[1] / 2, ylim[0] - 180, '~', fontsize=10, ha='center')

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
    ax[-1].text(width / 2 + 0.1, 1600, 'VISa/am', rotation=90, va='center', color='w',
                fontweight='bold', ha='center', size=5)
    ax[-1].text(width / 2 + 0.1, 900, 'CA1', rotation=90, va='center', color='w',
                fontweight='bold', ha='center', size=5)
    ax[-1].text(width / 2 + 0.1, 300, 'DG', rotation=90, va='center', color='w',
                fontweight='bold', ha='center', size=5)
    ax[-1].text(width / 2 + 0.1, -500, 'LP', rotation=90, va='center', color='w',
                fontweight='bold', ha='center', size=5)
    ax[-1].text(width / 2 + 0.1, -1500, 'PO', rotation=90, va='center', color='w',
                fontweight='bold', ha='center', size=5)
    ax[-1].set_axis_off()

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    cbar.ax.set_yticklabels([f'{levels[0]}', f'{levels[1]}'])
    cbar.set_label('Firing rate (spikes/s)', rotation=270, labelpad=-2)

    # Return the list of pids used in this figure
    return np.unique(df_filt['pid'])


def panel_permutation(ax, metrics, regions, labels, n_permut=10000, n_rec_per_lab=0,
                      n_rec_per_region=3, bh_correction=False, freeze=None):

    df_ins = load_dataframe(df_name='ins')
    df_filt = filter_recordings(df_ins, min_lab_region=n_rec_per_region, min_rec_lab=n_rec_per_lab,
                                min_neuron_region=2, recompute=False, n_trials=0, freeze=freeze)
    data = df_filt[df_filt['permute_include'] == 1]
    data['yield_per_channel'] = data['neuron_yield'] / data['n_channels']
    data.loc[data['lfp_power'] < -100000, 'lfp_power'] = np.nan

    results = pd.DataFrame()
    for i, metric in enumerate(metrics):
        print(f'Running permutation tests for metric {metric} ({i+1} of {len(metrics)})')
        for region in regions:
            # Select data for this region and metrics
            this_data = data.loc[data['region'] == region, metric].values
            this_labs = data.loc[data['region'] == region, 'institute'].values
            this_subjects = data.loc[data['region'] == region, 'subject'].values
            this_sessions = data.loc[data['region'] == region, 'eid'].values
            this_labs = this_labs[~np.isnan(this_data)]
            this_subjects = this_subjects[~np.isnan(this_data)]
            this_data = this_data[~np.isnan(this_data)]
            this_sessions = this_sessions[~np.isnan(this_data)]

            # Exclude data from labs that do not have enough recordings
            lab_names, this_n_labs = np.unique(this_labs, return_counts=True)
            excl_labs = lab_names[this_n_labs < n_rec_per_region]
            this_data = this_data[~np.isin(this_labs, excl_labs)]
            this_subjects = this_subjects[~np.isin(this_labs, excl_labs)]
            this_labs = this_labs[~np.isin(this_labs, excl_labs)]
            this_sessions = this_sessions[~np.isin(this_labs, excl_labs)]

            if PRINT_INFO:
                print(f'Figure 3 d: {metric}: {region}')
                print(f'N_inst: {len(np.unique(this_labs))}, N_sess: {len(np.unique(this_sessions))}, '
                      f'N_mice: {len(np.unique(this_subjects))}, N_cells: NA')


            # Do permutation test
            p = permut_test(this_data, metric=distribution_dist_approx_max, labels1=this_labs,
                            labels2=this_subjects, n_permut=n_permut, plot=False, n_cores=4)
            results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1], data={
                'metric': metric, 'region': region, 'p_value_permut': p})))

    for i, region in enumerate(regions):
        results.loc[results['region'] == region, 'region_number'] = i

    # Perform Benjamin-Hochman correction for multiple testing
    if bh_correction:
        _, results['p_value_permut'], _, _ = multipletests(results['p_value_permut'], 0.05,
                                                           method='fdr_bh')

    results_plot = results.pivot(index='region_number', columns='metric', values='p_value_permut')
    results_plot = results_plot.reindex(columns=metrics)
    results_plot = np.log10(results_plot)

    axin = inset_axes(ax, width="5%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)

    # Create colormap
    RdYlGn = cm.get_cmap('RdYlGn', 256)(np.linspace(0, 1, 800))


    color_array = np.vstack([np.tile(np.concatenate((to_rgb('darkviolet'), [1])), (200, 1)), RdYlGn])
    newcmp = ListedColormap(color_array)

    sns.heatmap(results_plot, cmap=newcmp, square=True,
                cbar=True, cbar_ax=axin,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', vmin=-2.5, vmax=np.log10(1), ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.01, 0.1, 1]))
    cbar.set_ticklabels([0.01, 0.1, 1])
    cbar.set_label('p-value (log scale)', rotation=270, labelpad=8)
    ax.set(xlabel='', ylabel='', xticks=np.arange(len(labels)) + 0.5, yticks=np.arange(len(regions)) + 0.5)
    regions[regions == 'PPC'] = 'VISa/am'
    ax.set_yticklabels(regions, va='center', rotation=0)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    return results, np.unique(data['pid'])


def panel_decoding(ax, qc='pass', region_decoding=True, bh_correction=False):

    # Load in data
    decode_df = load_dataframe(df_name=f'decode_{qc}')
    shuffle_df = load_dataframe(df_name=f'decode_shuf_{qc}')
    decode_df['accuracy'] = decode_df['accuracy']*100
    shuffle_df['accuracy_shuffle'] = shuffle_df['accuracy_shuffle']*100
    decode_regions_df = decode_df[decode_df['region'] == 'all']
    shuffle_regions_df = shuffle_df[shuffle_df['region'] == 'all']
    decode_df = decode_df[decode_df['region'] != 'all']
    shuffle_df = shuffle_df[shuffle_df['region'] != 'all']
    decode_df.loc[decode_df['region'] == 'PPC', 'region'] = 'VIS'
    shuffle_df.loc[shuffle_df['region'] == 'PPC', 'region'] = 'VIS'

    # Get p-values
    p_values = dict()
    for i, region in enumerate(decode_df['region']):
        p_values[region] = (np.sum(decode_df.loc[decode_df['region'] == region, 'accuracy'].values
                                  < shuffle_df.loc[shuffle_df['region'] == region, 'accuracy_shuffle'].values)
                            / shuffle_df.loc[shuffle_df['region'] == region, 'accuracy_shuffle'].shape[0])

    # Perform Benjamin-Hochman correction for multiple testing
    if bh_correction:
        _, p_values_corr, _, _ = multipletests(list(p_values.values()), 0.05, method='fdr_bh')
        for i, region in enumerate(list(p_values.keys())):
            p_values[region] = p_values_corr[i]

    if region_decoding:
        # Plot region decoding
        divider = make_axes_locatable(ax)
        ax_left = divider.append_axes("left", size='25%', pad='60%', sharey=ax)
        sns.violinplot(x='region', y='accuracy_shuffle', data=shuffle_regions_df, ax=ax_left,
                       color=[.7, .7, .7], linewidth=0, width=0.075)
        sns.scatterplot(x='region', y='accuracy', data=decode_regions_df, ax=ax_left, color='red',
                        marker='_', linewidth=1)
        ax_left.set(xlabel='', ylabel='Region decoding perf. (%)', xticks=[])
        ax_left.text(0, 80, '***', size=7, color='k', ha='center')

    # Plot decoding of lab per region
    sns.violinplot(x='region', y='accuracy_shuffle', data=shuffle_df, ax=ax, color=[.7, .7, .7],
                   linewidth=0)
    sns.scatterplot(x='region', y='accuracy', data=decode_df, ax=ax, color='red',
                    marker='_', legend=None, linewidth=1)

    # Plot significance star
    for i, region in enumerate(list(p_values.keys())):
        if p_values[region] < 0.001:
            ax.text(i, 50, '***', color='k', size=7, ha='center')
        elif p_values[region] < 0.01:
            ax.text(i, 50, '**', color='k', size=7, ha='center')
        elif p_values[region] < 0.05:
            ax.text(i, 50, '*', color='k', size=7, ha='center')

    # Settings
    ax.set(ylim=[0, 80], xlabel='', ylabel='Lab decoding perf. (%)', yticks=[0, 80], xlim=[-0.5, 4.5])
    sns.despine(trim=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax_left.spines['bottom'].set_visible(False)

    return p_values


def panel_example(ax, n_rec_per_lab=0, n_rec_per_region=3,
                  example_region='DG', example_metric='lfp_power', ylabel='LFP power in DG (db)',
                  ylim=None, yticks=None, despine=True, freeze=None):

    df_ins = load_dataframe(df_name='ins')
    df_filt = filter_recordings(df_ins, min_rec_lab=n_rec_per_lab, min_lab_region=n_rec_per_region,
                                min_neuron_region=2, recompute=False, freeze=freeze)
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt['yield_per_channel'] = df_filt['neuron_yield'] / df_filt['n_channels']
    df_filt.loc[df_filt['lfp_power'] < -100000, 'lfp_power'] = np.nan
    data = df_filt[df_filt['permute_include'] == 1]

    data_example = pd.DataFrame(data={
        'institute': data.loc[data['region'] == example_region, 'institute'],
        'lab_number': data.loc[data['region'] == example_region, 'lab_number'],
        'session': data.loc[data['region'] == example_region, 'eid'],
        'subject': data.loc[data['region'] == example_region, 'subject'],
        example_metric: data.loc[data['region'] == example_region, example_metric].values})
    data_example = data_example[~data_example[example_metric].isnull()]

    if PRINT_INFO:
        print(f'Figure 3 supp 4 {example_region}: {example_metric}')
        print(f'N_inst: {data_example.institute.nunique()}, N_sess: {data_example.session.nunique()}, '
              f'N_mice: {data_example.subject.nunique()}, N_cells: NA')

    data_example = data_example.sort_values('institute')
    cmap = []
    for i, inst in enumerate(data_example['institute'].unique()):
        cmap.append(lab_colors[inst])

    sns.swarmplot(data=data_example, x='institute', y=example_metric, palette=cmap, s=3, ax=ax)

    """
    # Plot lab means and overal mean
    ax_lines = sns.pointplot(x='institute', y=example_metric, data=data_example,
                             ci=0, join=False, estimator=np.mean, color='k',
                             markers="_", scale=1, ax=ax)
    plt.setp(ax_lines.collections, zorder=100, label="")
    ax.plot(np.arange(data_example['institute'].unique().shape[0]),
             [data_example[example_metric].mean()] * data_example['institute'].unique().shape[0],
             color='r', lw=1)
    """

    ax.set(ylabel=ylabel, xlabel='', xlim=[-.5, len(data_example['institute'].unique())])
    if ylim is not None:
        ax.set(ylim=ylim)
    if yticks is not None:
        ax.set(yticks=yticks)
    ax.set_xticklabels(data_example['institute'].unique(), rotation=90, ha='center')
    #ax.plot([-.5, len(data['institute'].unique()) + .5], [-165, -165], lw=0.5, color='k')
    #ax.plot([-0.5, -0.5], ax.get_ylim(),  lw=0.5, color='k')

    if despine:
        sns.despine(trim=True)
