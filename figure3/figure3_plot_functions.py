from reproducible_ephys_functions import filter_recordings, labs, BRAIN_REGIONS, query, get_insertions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figure3.figure3_functions import get_brain_boundaries, plot_probe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ibllib.atlas.regions import BrainRegions
from figure3.figure3_load_data import load_dataframe
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
from permutation_test import permut_test, distribution_dist_approx_max
from statsmodels.stats.multitest import multipletests

br = BrainRegions()

lab_number_map, institution_map, lab_colors = labs()


def panel_sankey(fig, ax, one, freeze=None):
    # Get all trajectories
    trajs_0, _ = get_insertions(level=0, freeze=freeze, as_dataframe=True)
    # Get trajs level 1
    trajs_1, ins_df_1 = get_insertions(level=1, freeze=freeze, as_dataframe=True)
    # Get trajs level 2
    trajs_2, ins_df_2 = get_insertions(level=2, freeze=freeze, as_dataframe=True)
    # Compute which PIDs are critical
    pids_crt = set(trajs_0.probe_insertion.values) - set(trajs_1.probe_insertion.values)

    # Remove critical PIDs that fail for Hardware failure of tracing missing --> left with Ephys issues
    remove_crt_histology = [
        '3443eceb-50b3-450e-b7c1-fc465a3bc84f',
        '553f8d56-b6e7-46bd-a146-ac43b8ec6de7',
        '6bfa4a99-bdfa-4e44-aa01-9c7eac7e253d',
        '82a42cdf-3140-427b-8ad0-0d504716c871',
        'dc6eea9b-a8fb-4151-b298-718f321e6968'
    ]

    remove_crt_hardware = [
        '143cff36-30b4-4b25-9d26-5a2dbe6f6fc2',
        '79bcfe47-33ed-4432-a598-66006b4cde56',
        '80624507-4be6-4689-92df-0e2c26c3faf3',
        'e7abb87f-4324-4c89-8a46-97ed4b40577e',
        'f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c',
        'fc626c12-bd1e-45c3-9434-4a7a8c81d7c0',
        'd8ff1218-75e1-4962-b920-98c40b9dea1a'
    ]

    pids_crt_ephysonly = pids_crt - set(remove_crt_histology) - set(remove_crt_hardware)


    #Sankey pot
    all_ins = trajs_0.shape[0]
    hw_crt = len(remove_crt_hardware)
    hist_crt = len(remove_crt_histology)
    ephys_crt = len(pids_crt_ephysonly)

    # ----
    df_drop_sankey = ins_df_1.copy()

    low_yield = df_drop_sankey.loc[(df_drop_sankey['low_yield'] == True)].shape[0]  # TODO do not know how to return indx from drop
    df_drop_sankey.drop(df_drop_sankey[df_drop_sankey['low_yield'] == True].index, inplace=True)

    high_noise = \
    df_drop_sankey.loc[(df_drop_sankey['high_noise'] == True) | (df_drop_sankey['high_lfp'] == True)].shape[0]
    df_drop_sankey.drop(
        df_drop_sankey[(df_drop_sankey['high_noise'] == True) | (df_drop_sankey['high_lfp'] == True)].index,
        inplace=True)

    behav = df_drop_sankey.loc[(df_drop_sankey['low_trials'] == True)].shape[0]
    df_drop_sankey.drop(df_drop_sankey[(df_drop_sankey['low_trials'] == True)].index, inplace=True)

    missed_target = df_drop_sankey.loc[(df_drop_sankey['missed_target'] == True)].shape[0]
    df_drop_sankey.drop(df_drop_sankey[(df_drop_sankey['missed_target'] == True)].index, inplace=True)
    assert df_drop_sankey.shape[0] == ins_df_2.shape[0]

    num_trajectories = [all_ins, -hw_crt, -hist_crt, -ephys_crt, -low_yield,
                        -high_noise, -behav, -missed_target, -ins_df_2.shape[0]]

    labels = ['All insertions', 'Hardware failure', 'Missing histology', 'Poor ephys', 'Low neural yield',
              'High noise', 'Poor behavior', 'Missed target', 'Data analysis']

    fig, ax = plt.subplots()
    sankey = Sankey(ax=ax, scale=0.005, offset=0.05, head_angle=90, shoulder=0.025, gap=0.2, radius=0.05)
    sankey.add(flows=num_trajectories,
               labels=labels,
               trunklength=0.7,
               orientations=[0, 1, 1, 1, 1, 1, 1, 1, 0],
               pathlengths=[0.08, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.2],
               facecolor=sns.color_palette('Pastel1')[1])
    diagrams = sankey.finish()

    # text font and positioning
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

    ax.axis('off')


def panel_probe_lfp(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000],
                    normalize=False, clim=[-190, -150], freeze=None):

    df_chns = load_dataframe(df_name='chns')
    df_filt = filter_recordings(min_rec_lab=n_rec_per_lab, min_neuron_region=0, freeze=freeze)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['institute', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('institute', group_keys=False).size()
    df_filt['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

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
    plt.figtext(0.24, 0.715, 'Berkeley', va="center", ha="center", size=7, color=lab_colors['Berkeley'])
    plt.figtext(0.34, 0.715, 'Champalimaud', va="center", ha="center", size=7, color=lab_colors['CCU'])
    plt.figtext(0.435, 0.715, 'CSHL', va="center", ha="center", size=7, color=lab_colors['CSHL (C)'])
    #plt.figtext(0.505, 0.715, '(Z)', va="center", ha="center", size=7, color=lab_colors['CSHL (Z)'])
    plt.figtext(0.515, 0.715, 'NYU', va="center", ha="center", size=7, color=lab_colors['NYU'])
    plt.figtext(0.57, 0.715, 'Princeton', va="center", ha="center", size=7, color=lab_colors['Princeton'])
    plt.figtext(0.63, 0.715, 'SWC', va="center", ha="center", size=7, color=lab_colors['SWC'])
    plt.figtext(0.735, 0.715, 'UCL', va="center", ha="center", size=7, color=lab_colors['UCL'])
    #plt.figtext(0.805, 0.715, '(H)', va="center", ha="center", size=7, color=lab_colors['UCL (H)'])
    plt.figtext(0.83, 0.715, 'UCLA', va="center", ha="center", size=7, color=lab_colors['UCLA'])
    plt.figtext(0.875, 0.715, 'UW', va="center", ha="center", size=7, color=lab_colors['UW'])

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="90%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    if normalize:
        cbar.ax.set_yticklabels(['10th\nperc.', '90th\nperc'])
    else:
        cbar.ax.set_yticklabels([f'{clim[0]}', f'{clim[1]}'])
    cbar.set_label('Power spectral density (dB)', rotation=270, labelpad=-5)


def panel_probe_neurons(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000],
                        freeze=None):

    df_chns = load_dataframe(df_name='chns')
    df_clust = load_dataframe(df_name='clust')

    df_filt = filter_recordings(min_rec_lab=n_rec_per_lab, min_neuron_region=0, freeze=freeze)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['institute', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('institute').size()
    df_filt['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

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
                      n_rec_per_region=3, bh_correction=False, freeze=None):

    df_ins = load_dataframe(df_name='ins')
    df_filt = filter_recordings(df_ins, min_lab_region=n_rec_per_region, min_rec_lab=n_rec_per_lab,
                                min_neuron_region=2, recompute=True, freeze=freeze)
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
    ax.set_yticklabels(regions, va='center', rotation=0)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    return results


def panel_decoding(ax, qc='pass', region_decoding=True):

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

    # Correct for multiple comparisons
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
        ax_left.text(0, 75, '***', size=7, color='k', ha='center')

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
    ax.set(ylim=[0, 80], xlabel='', ylabel='Lab decoding perf. (%)', yticks=[0, 75], xlim=[-0.5, 4.5])
    sns.despine(trim=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax_left.spines['bottom'].set_visible(False)

    return p_values
