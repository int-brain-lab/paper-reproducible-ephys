from reproducible_ephys_functions import filter_recordings, labs, BRAIN_REGIONS, query, get_insertions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figure3.figure3_functions import get_brain_boundaries, plot_probe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ibllib.atlas.regions import BrainRegions
from figure3.figure3_load_data import load_dataframe
import seaborn as sns
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
from permutation_test import permut_test, distribution_dist_approx
from statsmodels.stats.multitest import multipletests

br = BrainRegions()

lab_number_map, institution_map, lab_colors = labs()

def panel_sankey(fig, ax, one):

    # Get number of recordings after freeze cutoff date
    after_freeze = len(get_insertions(one=one)) - len(get_insertions(one=one, freeze=None))

    # Get number of failed recordings (CRITICAL)
    crit = (len(query(behavior=False, n_trials=0, resolved=False, min_regions=0, exclude_critical=False, one=one))
            - len(query(behavior=False, n_trials=0, resolved=False, min_regions=0, exclude_critical=True, one=one))
            - after_freeze)

    # Get total number of insertions
    all_ins = (len(query(behavior=False, n_trials=0, resolved=True, min_regions=0, exclude_critical=True, one=one))
               + crit - after_freeze)

    # Dropout due to targeting
    target = all_ins - crit - len(query(behavior=False, n_trials=0, resolved=True,
                                        min_regions=2, exclude_critical=True, one=one))

    # Dropout due to behavior
    behav = all_ins - crit - target - len(query(behavior=False, n_trials=400, resolved=True,
                                                min_regions=2, exclude_critical=True, one=one))

    # Get secondary QC
    df_filt = filter_recordings(min_rec_lab=0, min_neuron_region=0)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    low_yield = df_filt['low_yield'].sum()
    noise = (df_filt[df_filt['high_lfp'] & ~df_filt['low_yield']].shape[0]
             + df_filt[df_filt['high_noise'] & ~df_filt['low_yield']].shape[0])
    artifacts = df_filt[(df_filt['artifacts'] & ~df_filt['low_yield'] & ~df_filt['high_noise']
                        & ~df_filt['high_lfp'])].shape[0]

    # Recordigns left
    rec_left = all_ins - crit - target - behav - low_yield - noise - artifacts

    #fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=400)
    ax.axis('off')

    #currently hardcoded to match Steven & Guido analyses;
    #todo: finalize numbers and match with above code
    num_trajectories = [all_ins, -crit, -target, -behav, -low_yield, -(noise+artifacts), -rec_left]

    # Sankey plot
    sankey = Sankey(ax=ax, scale=0.005, offset=0.1, head_angle=90, shoulder=0.025, gap=0.5, radius=0.05)
    sankey.add(flows=num_trajectories,
               labels=['All insertions',
                       'Recording failure',
                       'Off target',
                       'Too few trials',
                       'Low yield',
                       'Noise/artifacts',
                       'Data analysis'],
               trunklength=0.8,
               orientations=[0, 1, 1, 1, 1, 1, 0],
               pathlengths=[0.08, 0.3, 0.15, 0.15, 0.1, 0.08, 0.4],
               facecolor = sns.color_palette('Pastel1')[1])
    diagrams = sankey.finish()

    #text font and positioning
    for text in diagrams[0].texts:
            text.set_fontsize('7')


    text = diagrams[0].texts[0]
    xy = text.get_position()
    text.set_position((xy[0] - 0.3, xy[1]))
    text.set_weight('bold')

    text = diagrams[0].texts[-1]
    xy = text.get_position()
    text.set_position((xy[0] + 0.2, xy[1]))
    text.set_weight('bold')


def panel_probe_lfp(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000],
                    normalize=False, clim=[-190, -150]):

    df_chns = load_dataframe(df_name='chns')
    df_filt = filter_recordings(min_rec_lab=n_rec_per_lab, min_neuron_region=0)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['institute', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('institute').size()
    df_filt['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

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

        ax[iR].set_title(data['recording'] + 1,
                         color=lab_colors[data['institute']])
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
    plt.figtext(0.245, 0.715, 'Berkeley', va="center", ha="center", size=7, color=lab_colors['Berkeley'])
    plt.figtext(0.345, 0.715, 'Champalimaud', va="center", ha="center", size=7, color=lab_colors['CCU'])
    plt.figtext(0.45, 0.715, 'CSHL (C)', va="center", ha="center", size=7, color=lab_colors['CSHL (C)'])
    plt.figtext(0.495, 0.715, '(Z)', va="center", ha="center", size=7, color=lab_colors['CSHL (Z)'])
    plt.figtext(0.56, 0.715, 'NYU', va="center", ha="center", size=7, color=lab_colors['NYU'])
    plt.figtext(0.64, 0.715, 'Princeton', va="center", ha="center", size=7, color=lab_colors['Princeton'])
    plt.figtext(0.71, 0.715, 'SWC', va="center", ha="center", size=7, color=lab_colors['SWC'])
    plt.figtext(0.785, 0.715, 'UCL', va="center", ha="center", size=7, color=lab_colors['UCL'])
    plt.figtext(0.85, 0.715, 'UCLA', va="center", ha="center", size=7, color=lab_colors['UCLA'])

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

    df_filt = filter_recordings(min_rec_lab=n_rec_per_lab, min_neuron_region=0)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['institute', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('institute').size()
    df_filt['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

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
        ax[iR].images.append(im)
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

        ax[iR].set_title(data['recording'] + 1,
                         color=lab_colors[data['institute']])

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
    df_filt = filter_recordings(df_ins, min_rec_lab=n_rec_per_lab, min_lab_region=n_rec_per_region,
                                min_neuron_region=2, recompute=False)
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt['yield_per_channel'] = df_filt['neuron_yield'] / df_filt['n_channels']
    df_filt.loc[df_filt['lfp_power'] < -100000, 'lfp_power'] = np.nan
    data = df_filt[df_filt['permute_include'] == 1]

    data_example = pd.DataFrame(data={
        'institute': data.loc[data['region'] == example_region, 'institute'],
        'lab_number': data.loc[data['region'] == example_region, 'lab_number'],
        example_metric: data.loc[data['region'] == example_region, example_metric].values})
    data_example = data_example[~data_example[example_metric].isnull()]

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



def panel_permutation(ax, metrics, regions, labels, n_permut=10000, n_rec_per_lab=0,
                      n_rec_per_region=3):

    df_ins = load_dataframe(df_name='ins')
    df_filt = filter_recordings(df_ins, min_lab_region=n_rec_per_region, min_rec_lab=n_rec_per_lab,
                                min_neuron_region=2, recompute=False)
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
            this_labs = this_labs[~np.isnan(this_data)]
            this_subjects = this_subjects[~np.isnan(this_data)]
            this_data = this_data[~np.isnan(this_data)]

            # Exclude data from labs that do not have enough recordings
            lab_names, this_n_labs = np.unique(this_labs, return_counts=True)
            excl_labs = lab_names[this_n_labs < n_rec_per_region]
            this_data = this_data[~np.isin(this_labs, excl_labs)]
            this_subjects = this_subjects[~np.isin(this_labs, excl_labs)]
            this_labs = this_labs[~np.isin(this_labs, excl_labs)]

            # Do permutation test
            p = permut_test(this_data, metric=distribution_dist_approx, labels1=this_labs,
                            labels2=this_subjects, n_permut=n_permut, plot=False)
            results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1], data={
                'metric': metric, 'region': region, 'p_value_permut': p})))

    for i, region in enumerate(regions):
        results.loc[results['region'] == region, 'region_number'] = i

    # Perform correction for multiple testing
    _, results['p_value_permut'], _, _ = multipletests(results['p_value_permut'], 0.05, method='fdr_bh')

    results_plot = results.pivot(index='region_number', columns='metric', values='p_value_permut')
    results_plot = results_plot.reindex(columns=metrics)
    results_plot = np.log10(results_plot)

    axin = inset_axes(ax, width="5%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
    # cmap = sns.color_palette('viridis_r', n_colors=20)
    # cmap[0] = [1, 0, 0]
    sns.heatmap(results_plot, cmap='RdYlGn', square=True,
                cbar=True, cbar_ax=axin,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', vmin=-1.5, vmax=np.log10(1), ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.05, 0.1, 0.2, 0.4, 0.8]))
    cbar.set_ticklabels([0.05, 0.1, 0.2, 0.4, 0.8])
    cbar.set_label('log p-value', rotation=270, labelpad=8)
    ax.set(xlabel='', ylabel='', xticks=np.arange(len(labels)) + 0.5, yticks=np.arange(len(regions)) + 0.5)
    ax.set_yticklabels(regions, va='center', rotation=0)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    return results


def panel_decoding(ax, qc=True):

    # Load in data
    if qc:
        decode_df = load_dataframe(df_name='decode')
        shuffle_df = load_dataframe(df_name='decode_shuf')
    else:
        decode_df = load_dataframe(df_name='decode_no_qc')
        shuffle_df = load_dataframe(df_name='decode_shuf_no_qc')
    decode_df['accuracy'] = decode_df['accuracy']*100
    shuffle_df['accuracy_shuffle'] = shuffle_df['accuracy_shuffle']*100
    decode_regions_df = decode_df[decode_df['region'] == 'all']
    shuffle_regions_df = shuffle_df[shuffle_df['region'] == 'all']
    decode_df = decode_df[decode_df['region'] != 'all']
    shuffle_df = shuffle_df[shuffle_df['region'] != 'all']

    # Get p-values
    p_values = dict()
    for i, region in enumerate(decode_df['region']):
        p_values[region] = (np.sum(decode_df.loc[decode_df['region'] == region, 'accuracy'].values
                                  < shuffle_df.loc[shuffle_df['region'] == region, 'accuracy_shuffle'].values)
                            / shuffle_df.loc[shuffle_df['region'] == region, 'accuracy_shuffle'].shape[0])

    # Correct for multiple comparisons
    _, p_values_corr, _, _ = multipletests(list(p_values.values()), 0.05, method='fdr_bh')
    for i, region in enumerate(list(p_values.keys())):
        p_values[region] = p_values_corr[i]

    # Plot
    divider = make_axes_locatable(ax)
    ax_left = divider.append_axes("left", size='25%', pad='60%', sharey=ax)

    # Plot region decoding
    sns.violinplot(x='region', y='accuracy_shuffle', data=shuffle_regions_df, ax=ax_left, color=[.7, .7, .7])
    sns.swarmplot(x='region', y='accuracy', data=decode_regions_df, ax=ax_left, color='red', size=4)
    ax_left.set(xlabel='', ylabel='Region decoding perf. (%)', xticks=[])
    ax_left.text(0, 75, '***', size=10, color='k', ha='center')

    # Plot decoding of lab per region
    sns.violinplot(x='region', y='accuracy_shuffle', data=shuffle_df, ax=ax, color=[.7, .7, .7])
    sns.swarmplot(x='region', y='accuracy', data=decode_df, ax=ax, color='red', size=4)

    # Plot significance star
    for i, region in enumerate(list(p_values.keys())):
        if p_values[region] < 0.001:
            ax.text(i, 50, '***', color='k', size=10, ha='center')
        elif p_values[region] < 0.01:
            ax.text(i, 50, '**', color='k', size=10, ha='center')
        elif p_values[region] < 0.05:
            ax.text(i, 50, '*', color='k', size=10, ha='center')

    # Settings
    ax.set(ylim=[0, 80], xlabel='', ylabel='Lab decoding perf. (%)', yticks=[0, 75])
    sns.despine(trim=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax_left.spines['bottom'].set_visible(False)

    return p_values
