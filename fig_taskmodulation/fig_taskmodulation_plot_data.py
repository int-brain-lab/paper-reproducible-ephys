import matplotlib.pyplot as plt
import numpy as np
import figrid as fg
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs, save_figure_path, figure_style
from fig_taskmodulation.fig_taskmodulation_load_data import load_data, load_dataframe
from fig_taskmodulation.fig_taskmodulation_plot_functions import plot_raster_and_psth, plot_raster_and_psth_LvsR
import seaborn as sns
import pandas as pd
from statsmodels.stats.multitest import multipletests
import pickle
from permutation_test import permut_test, distribution_dist_approx, shuffle_labels, distribution_dist_approx_max
from matplotlib.transforms import Bbox


lab_number_map, institution_map, lab_colors = labs()
fig_path = save_figure_path(figure='fig_taskmodulation')

# tests = {'trial': 'Trial',
#          'start_to_move': 'Pre move (TW)',
#          'post_stim': 'Post stim',
#          'pre_move': 'Pre move',
#          'pre_move_lr': 'Move LvR',
#          'post_move': 'Post move',
#          'post_reward': 'Post reward',
#          'avg_ff_post_move': 'FanoFactor'}

# tests = {'trial': 'Trial',
#           'start_to_move': 'Reaction period',
#           'post_stim': 'Post-stimulus',
#           'pre_move': 'Pre-movement',
#           'pre_move_lr': 'L vs. R pre-movement',
#           'post_move': 'Post-movement',
#           'post_reward': 'Post-reward',
#           'avg_ff_post_move': 'Fano Factor'}

# shortened_tests = {'trial': 'Trial',
#                    'start_to_move': 'Reaction',
#                    'post_stim': 'Post-stim',
#                    'pre_move': 'Pre-move',
#                    'pre_move_lr': 'LvR move',
#                    'post_move': 'Post-move',
#                    'post_reward': 'Post-rew',
#                    'avg_ff_post_move': 'FF'}

#Renamed & remove 'Trial':
tests = {#'trial': 'Trial (first 400 ms)',
          'start_to_move': 'Late reaction period',
          'post_stim': 'Stimulus',
          'pre_move': 'Movement initiation',
          'pre_move_lr': 'L vs. R pre-movement',
          'post_move': 'Movement period (250 ms)',
          'post_reward': 'Reward',
          'avg_ff_post_move': 'Fano Factor'}

shortened_tests = {#'trial': 'Trial (first 400 ms)',
                   'start_to_move': 'Reaction period',
                   'post_stim': 'Stimulus',
                   'pre_move': 'Move. initiation',
                   'pre_move_lr': 'L vs R move.',
                   'post_move': 'Move. (250 ms)',
                   'post_reward': 'Reward',
                   'avg_ff_post_move': 'Fano Factor'}


def plot_main_figure():
    DPI = 400  # if the figure is too big on your screen, lower this number
    figure_style()
    fig = plt.figure(figsize=(7, 10.5), dpi=DPI)  # full width figure is 7 inches
    # ax = {'panel_A_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.323], yspan=[0., 0.15],
    #                                          wspace=0.3),
    #       'panel_A_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.323], yspan=[0.15, 0.3],
    #                                          wspace=0.3),
    #       'panel_B': fg.place_axes_on_grid(fig, xspan=[0.423, 0.671], yspan=[0., 0.3],
    #                                        wspace=0.3),
    #       'panel_C': fg.place_axes_on_grid(fig, xspan=[0.751, 1.], yspan=[0., 0.3],
    #                                        wspace=0.3),
    #       'panel_D_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.27375], yspan=[0.38, 0.65],
    #                                          wspace=0.3),
    #       'panel_D_2': fg.place_axes_on_grid(fig, xspan=[0.28375, 0.4825], yspan=[0.38, 0.65],
    #                                          wspace=0.3),
    #       'panel_D_3': fg.place_axes_on_grid(fig, xspan=[0.4925, 0.69125], yspan=[0.38, 0.65],
    #                                          wspace=0.3),
    #       'panel_D_4': fg.place_axes_on_grid(fig, xspan=[0.70125, 0.9], yspan=[0.38, 0.65],
    #                                          wspace=0.3)}
    ax = {'panel_A_1': fg.place_axes_on_grid(fig, xspan=[0.08, 0.288], yspan=[0.045, 0.13],
                                             wspace=0.3),
          'panel_A_2': fg.place_axes_on_grid(fig, xspan=[0.08, 0.288], yspan=[0.14, 0.26],
                                             wspace=0.3),
          'panel_B': fg.place_axes_on_grid(fig, xspan=[0.388, 0.631], yspan=[0.045, 0.26],
                                           wspace=0.3),
          'panel_C': fg.place_axes_on_grid(fig, xspan=[0.741, 1.], yspan=[0.045, 0.26],
                                           wspace=0.3),
          'panel_D_1': fg.place_axes_on_grid(fig, xspan=[0.075,  0.27375], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_D_2': fg.place_axes_on_grid(fig, xspan=[0.28375, 0.4825], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_D_3': fg.place_axes_on_grid(fig, xspan=[0.4925, 0.69125], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_D_4': fg.place_axes_on_grid(fig, xspan=[0.70125, .9], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_E_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.71, 0.76],
                                             wspace=0.3),
          'panel_E_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.77, 0.82],
                                             wspace=0.3),
          'panel_E_3': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.83, 0.88],
                                             wspace=0.3),
          'panel_E_4': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.89, 0.94],
                                             wspace=0.3),
          'panel_E_5': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.95, 1.],
                                             wspace=0.3),
          'panel_F_1': fg.place_axes_on_grid(fig, xspan=[0.55, 0.66], yspan=[0.69, .8],
                                             wspace=0.3),
          'panel_F_3': fg.place_axes_on_grid(fig, xspan=[0.68, 1.], yspan=[0.69, .8],
                                             wspace=0.3),
          'panel_F_2': fg.place_axes_on_grid(fig, xspan=[0.55, 1.], yspan=[0.8, .97],
                                             wspace=0.3)}
          # 'panel_F': fg.place_axes_on_grid(fig, xspan=[0.08, .99], yspan=[0.75, .91],
          #                                  wspace=0.3)}

    #plot_panel_single_neuron(ax=[ax['panel_A_1'], ax['panel_A_2']], save=False)
    plot_panel_single_neuron_LvsR(ax=[ax['panel_A_1'], ax['panel_A_2']], save=False)
    plot_panel_single_subject(ax=ax['panel_B'], save=False)
    plot_panel_task_modulated_neurons(specific_tests=['pre_move'],
                                      ax=[ax['panel_E_1'], ax['panel_E_2'], ax['panel_E_3'], ax['panel_E_4'], ax['panel_E_5']],
                                      save=False)
    plot_panel_power_analysis(ax=ax['panel_F_3'], ax2=ax['panel_F_1'])
    plot_panel_permutation(ax=ax['panel_F_2'])

    # we have to find out max and min neurons here now, because plots are split
    df = load_dataframe()
    df_filt = filter_recordings(df, freeze='release_2022_11')
    df_filt = df_filt[df_filt['include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby('region')
    max_neurons = 0
    min_neurons = 1000000
    for iR, reg in enumerate(BRAIN_REGIONS):
        df_reg = df_filt_reg.get_group(reg)
        df_reg_subj = df_reg.groupby('subject')
        for subj in df_reg_subj.groups.keys():
            subj_idx = df_reg_subj.groups[subj]
            max_neurons = max(max_neurons, subj_idx.shape[0])
            min_neurons = min(min_neurons, subj_idx.shape[0])
    D_regions = [reg for reg in BRAIN_REGIONS if reg != 'LP']
    plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons, ax=[ax['panel_C']], save=False, plotted_regions=['LP'])
    plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons, ax=[ax['panel_D_1'], ax['panel_D_2'], ax['panel_D_3'], ax['panel_D_4']], save=False, plotted_regions=D_regions)

    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0.005, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'b', 'xpos': 0.305, 'ypos': 0.005, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'c', 'xpos': 0.66, 'ypos': 0.005, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'd', 'xpos': 0, 'ypos': 0.34, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'e', 'xpos': 0, 'ypos': 0.66, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'f', 'xpos': 0.538, 'ypos': 0.66, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'g', 'xpos': 0.662, 'ypos': 0.66, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'h', 'xpos': 0.538, 'ypos': 0.82, 'fontsize': 10, 'weight': 'bold'}]

    fg.add_labels(fig, labels)
    print(f'Saving figures to {fig_path}')
    plt.savefig(fig_path.joinpath('fig_taskmodulation_combined.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('fig_taskmodulation_combined.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_panel_single_neuron(ax=None, save=True):
    # Code to plot figure similar to figure 4a; plots separately for each contrast

    pid = 'f26a6ab1-7e37-4f8d-bb50-295c056e1062'
    neuron = 241 #386
    align_event = 'move'
    params = {'smoothing': 'sliding',
              'fr_bin_size': 0.06,
              'event_epoch': [-0.2, 0.2], #[-0.3, 0.22],
              'slide_kwargs_fr': {'n_win': 3, 'causal': 1}}

    # neuron = 241 #323 #265 #144 #614
    # pid = 'a12c8ae8-d5ad-4d15-b805-436ad23e5ad1' #'36362f75-96d8-4ed4-a728-5e72284d0995'#'31f3e083-a324-4b88-b0a4-7788ec37b191' #'ce397420-3cd2-4a55-8fd1-5e28321981f4'  # SWC_054
    #side = 'right' #'left' #'all'
    feedback = 'correct' #'all'

    ax = plot_raster_and_psth(pid, neuron, align_event=align_event, feedback=feedback,
                              labelsize=16, ax=ax, contrasts=(1, 0.25, 0.125, 0.0625), **params) #excluding 0 contrasts
    ax[0].set_title('Example LP neuron', loc='left')

    if save:
        plt.savefig(fig_path.joinpath(f'fig_taskmodulation_{pid}_neuron{neuron}_align_{align_event}.png'))

    # Need to put legend for colorbar/side


def plot_panel_single_neuron_LvsR(ax=None, save=True):
    # Does not distinguish between contrasts, but distinguishes by side

    pid = 'f26a6ab1-7e37-4f8d-bb50-295c056e1062'
    neuron = 241  # 386
    align_event = 'move'
    params = {'smoothing': 'sliding',
              'fr_bin_size': 0.06,
              'event_epoch': [-0.2, 0.2],  # [-0.3, 0.22],
              'slide_kwargs_fr': {'n_win': 3, 'causal': 1}}

    # neuron = 241 #323 #265 #144 #614
    # pid = 'a12c8ae8-d5ad-4d15-b805-436ad23e5ad1' #'36362f75-96d8-4ed4-a728-5e72284d0995'#'31f3e083-a324-4b88-b0a4-7788ec37b191' #'ce397420-3cd2-4a55-8fd1-5e28321981f4'  # SWC_054
    side = 'right' #'left' #'all'
    feedback = 'correct' #'all'

    ax = plot_raster_and_psth_LvsR(pid, neuron, align_event=align_event, side=side, feedback=feedback,
                              labelsize=16, ax=ax, **params) #fr_bin_size=0.06, zero_line_c='g',

    # ax = plot_raster_and_psth(pid, neuron, align_event=align_event, side='left', ax=ax, **params)
    # ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.2, 0.2], fr_bin_size=0.06, align_event=align_event, side=side,
    #                           feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 3, 'causal': 1},
    #                           zero_line_c='g', labelsize=16, ax=ax)

    if save:
        plt.savefig(fig_path.joinpath(f'fig_taskmodulation_{pid}_neuron{neuron}_align_{align_event}.png'))

    #ax[0].set_title(f'Contrast: {side}, {feedback} choices', loc='left')
    #ax[0].set_title(f'{side} stim., {feedback} choices', loc='left')
    ax[0].set_title('Example LP neuron', loc='left')
    #Need to put legend for colorbar/contrasts


def plot_panel_single_subject(event='move', norm='subtract', smoothing='sliding', ax=None, save=True):
    # Code to plot figure similar to figure 4b
    df = load_dataframe()
    data = load_data(event=event, norm=norm, smoothing=smoothing)

    df_filt = filter_recordings(df, freeze='release_2022_11')
    all_frs_l = data['all_frs_l'][df_filt['include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['include'] == 1]
    all_frs_l_std = data['all_frs_l_std'][df_filt['include'] == 1]
    all_frs_r_std = data['all_frs_r_std'][df_filt['include'] == 1]
    df_filt = df_filt[df_filt['include'] == 1].reset_index()

    # Example to get similar plot to figure 4b
    if ax is None:
        fig, ax = plt.subplots()
    subject = 'DY_018' #'SWC_054'
    region = 'LP'
    idx = df_filt.loc[(df_filt['region'] == region) & (df_filt['subject'] == subject)].index
    # print(df_filt.loc[(df['region'] == region) & (df['subject'] == subject)].index)
    # print(df_filt.loc[(df['region'] == region) & (df['subject'] == subject)].institute)
    # a = df_filt.groupby(['subject', 'institute'])
    # print(a.groups.keys())
    # print(idx[0])
    # print(idx)
    # print(np.unique(df_filt.loc[idx].institute))
    lab = df_filt.loc[idx[0]].institute
    # print(subject)
    # print(lab)
    # quit()
    time = data['time']
    # To easily switch between sides for plotting:
    all_frs_side = all_frs_r  # all_frs_l #
    all_frs_side_std = all_frs_r_std  # all_frs_l_std #

    propagated_error = np.zeros_like(all_frs_side[idx][0])
    for fr, fr_std in zip(all_frs_side[idx], all_frs_side_std[idx]):
        ax.plot(time, fr, 'k')
        propagated_error += fr_std ** 2
        ax.fill_between(time, fr - fr_std, fr + fr_std, color='k', alpha=0.25)

    fr_mean = np.mean(all_frs_side[idx], axis=0)
    fr_std = np.std(all_frs_side[idx], axis=0)
    ax.plot(time, fr_mean, c=lab_colors[lab], lw=1.5)
    propagated_error = np.sqrt(propagated_error) / idx.shape[0]
    ax.fill_between(time, fr_mean - propagated_error, fr_mean + propagated_error, color=lab_colors[lab], alpha=0.25)
    ax.axvline(0, color='k', ls='--')

    # ax.set_title("Single mouse, {}".format(region))
    ax.set_xlabel("Time from movement onset (s)")
    ax.set_ylabel("Baselined firing rate (sp/s)", labelpad=-0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(left=time[0], right=time[-1])
    ax.set_xticks([-0.15, 0, 0.15], [-0.15, 0, 0.15]) #change this later
    # ax.set_xlim(left=-0.2, right=0.2) #change this later
    # sns.despine(trim=True, ax=ax)

    if save:
        plt.savefig(fig_path.joinpath('fig_taskmodulation_example_subject.png'))

    ax.set_title('Example recording in LP', loc='left')


def plot_panel_all_subjects(max_neurons, min_neurons, ax=None, save=True, plotted_regions=BRAIN_REGIONS):
    # Code to plot figure similar to figure 4c
    df = load_dataframe()
    data = load_data(event='move', norm='subtract', smoothing='sliding')

    df_filt = filter_recordings(df, freeze='release_2022_11')
    all_frs_l = data['all_frs_l'][df_filt['include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['include'] == 1]
    all_frs_l_std = data['all_frs_l_std'][df_filt['include'] == 1]
    all_frs_r_std = data['all_frs_r_std'][df_filt['include'] == 1]
    df_filt = df_filt[df_filt['include'] == 1].reset_index()

    # Example to get similar plot to figure 4c
    if ax is None:
        fig, ax = plt.subplots(1, len(plotted_regions))
    df_filt_reg = df_filt.groupby('region')

    min_lw = 0.5
    max_lw = 2

    print("max neurons: {}; min neurons: {}".format(max_neurons, min_neurons))

    all_present_labs = []
    for iR, reg in enumerate(plotted_regions):
        df_reg = df_filt_reg.get_group(reg)
        df_reg_subj = df_reg.groupby('subject')
        for subj in df_reg_subj.groups.keys():
            df_subj = df_reg_subj.get_group(subj)
            subj_idx = df_reg_subj.groups[subj]

            # Select L vs R side:
            # frs_subj = all_frs_l[subj_idx, :]
            frs_subj = all_frs_r[subj_idx, :]

            if df_subj.iloc[0]['institute'] not in all_present_labs:
                all_present_labs.append(df_subj.iloc[0]['institute'])
            ax[iR].plot(data['time'], np.mean(frs_subj, axis=0), c=lab_colors[df_subj.iloc[0]['institute']],
                        lw=min_lw + ((subj_idx.shape[0] - min_neurons) / (max_neurons - min_neurons)) * max_lw,
                        alpha=0.8)
        ax[iR].set_ylim(bottom=-9, top=21.5)
        ax[iR].set_yticks([-5, 0, 5, 10, 15, 20])
        ax[iR].axvline(0, color='k', ls='--')
        ax[iR].spines["right"].set_visible(False)
        ax[iR].spines["top"].set_visible(False)
        ax[iR].set_xlim(left=data['time'][0], right=data['time'][-1])
        ax[iR].set_xticks([-0.15, 0, 0.15], [-0.15, 0, 0.15])  # change this later
        # ax[iR].set_xlim(left=-0.2, right=0.2)  # change this later
        # sns.despine(trim=True, ax=ax[iR])
        if iR >= 1:
            ax[iR].set_yticklabels([])
        else:
            ax[iR].set_ylabel("Baselined firing rate (sp/s)")
            # ax[iR].set_title('Recordings from all labs', loc='left')
            # if len(plotted_regions) != 1:
            #     ax[iR].set_ylabel("Baselined firing rate (sp/s)")
            #     ax[iR].set_xlabel("Time (s)")
        # ax[iR].set_title(reg)

        if len(plotted_regions) == 1:
            ax[iR].set_title('Recording averages in LP', loc='left')
        else:
            ax[iR].set_title(reg)

        if iR == 1 or len(plotted_regions) == 1:
            ax[iR].set_xlabel("Time from movement onset (s)")

        if iR == len(plotted_regions) - 1 and len(plotted_regions) != 1:
            # this is a hack for the legend
            for lab in all_present_labs:
                ax[iR].plot(data['time'], np.zeros_like(data['time']) - 100, c=lab_colors[lab], label=lab)
            leg = ax[iR].legend(frameon=False, bbox_to_anchor=(1, 1), labelcolor='linecolor', handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)

    if save:
        plt.savefig(fig_path.joinpath('fig_taskmodulation_all_subjects.png'))


def plot_panel_task_modulated_neurons(specific_tests=None, ax=None, save=True):

    # load dataframe from prev fig. 5 (To be combined with new Fig 4)
    df = load_dataframe()
    df_filt = filter_recordings(df, freeze='release_2022_11')
    df_filt = df_filt[df_filt['include'] == 1]

    # Group data frame by region
    df_region = df_filt.groupby('region')

    names = tests.keys() if specific_tests is None else specific_tests
    # FIGURE 5c and supplementary figures
    for test in names:
        for i, br in enumerate(BRAIN_REGIONS):
            df_br = df_region.get_group(br)

            df_inst = df_br.groupby(['subject', 'institute'], as_index=False)
            vals = df_inst[test].mean().sort_values('institute')
            colors = [lab_colors[col] for col in vals['institute'].values]
            if ax is None:
                plt.subplot(len(BRAIN_REGIONS), 1, i + 1)
                plt.bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
                plt.ylim(bottom=0, top=1)
                plt.ylabel(br)
                plt.yticks([0, 1], [0, 1])
                plt.xticks([])
                sns.despine()
                if i == 4:
                    plt.xlabel('Mice')
                elif i == 0:
                    plt.title('% modulated neurons', loc='left')
            else:
                ax[i].bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
                ax[i].set_ylim(bottom=0, top=1)
                ax[i].set_ylabel(br, labelpad=-5)
                ax[i].set_yticks([0, 1], [0, 1])
                ax[i].set_xticks([])
                sns.despine()
                if i == 4:
                    ax[i].set_xlabel('Mice')
                elif i == 0:
                    ax[i].set_title('% modulated neurons (Pre-move.)', loc='left')
        if specific_tests is None:
            plt.suptitle(tests[test], size=22)
        if save:
            plt.savefig(fig_path.joinpath(test))


def plot_panel_permutation(ax=None, recompute=False, n_permut=20000, qc='pass', n_cores=8):
    """
    qc can be "pass" (only include recordings that pass QC)
    "high_noise": add the recordings with high noise
    "low_yield": add low yield recordings
    "missed_target": add recordings that missed the target regions
    "artifacts": add recordings with artifacts
    "low_trials": add recordings with < 400 trials
    "high_lfp": add recordings with high LFP power
    "all": add all recordings regardless of QC
    """
    # load dataframe from prev fig. 5 (To be combined with new Fig 4)
    # Prev Figure 5d permutation tests
    df = load_dataframe()
    df_filt = filter_recordings(df, recompute=True, min_lab_region=2, min_rec_lab=0, min_neuron_region=2, freeze='release_2022_11')
    if qc == 'pass':
        df_filt = df_filt[df_filt['permute_include'] == 1]
    elif qc != 'all':
        df_filt = df_filt[(df_filt['permute_include'] == 1) | (df_filt[qc] == 1)]

    test_names = [shortened_tests[test] for test in tests.keys()]
    if recompute:
        df_filt_reg = df_filt.groupby('region')
        results = pd.DataFrame()

        for test in tests.keys():
            for reg in BRAIN_REGIONS:
                df_reg = df_filt_reg.get_group(reg)
                # vals = df_reg.groupby(['institute', 'subject'])[test].mean()
                # labs = vals.index.get_level_values('institute')
                # subjects = vals.index.get_level_values('subject')
                # data = vals.values
                if test == 'avg_ff_post_move':
                    data = df_reg[test].values
                else:
                    data = df_reg['mean_fr_diff_{}'.format(test)].values
                labs = df_reg['institute'].values
                subjects = df_reg['subject'].values

                labs = labs[~np.isnan(data)]
                subjects = subjects[~np.isnan(data)]
                data = data[~np.isnan(data)]
                # lab_names, this_n_labs = np.unique(labs, return_counts=True)  # what is this for?

                print(".", end='')
                p = permut_test(data, metric=distribution_dist_approx_max, labels1=labs,
                                labels2=subjects, shuffling='labels1_based_on_2', n_cores=n_cores, n_permut=n_permut)

                # print(p)
                # if p > 0.05:
                #     return data, labs, subjects
                results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1],
                                                           data={'test': test, 'region': reg, 'p_value_permut': p})))

        pickle.dump(results.p_value_permut.values, open("p_values_new_max_metric", 'wb'))

    shape = (len(tests.keys()), len(BRAIN_REGIONS))
    print('shortening the trials test away, yes?')
    input()
    p_vals = pickle.load(open("p_values_new_max_metric", 'rb'))[5:]
    print(p_vals)
    print(np.sort(p_vals))
    # _, corrected_p_vals, _, _ = multipletests(results.p_value_permut.values, 0.05, method='fdr_bh')
    p_vals = p_vals.reshape(shape)
    # corrected_p_vals = results.p_value_permut.values.reshape(shape)

    ax = sns.heatmap(np.log10(p_vals.T), cmap='RdYlGn', square=True,
                     cbar=True, annot=False, annot_kws={"size": 12}, ax=ax,
                     linewidths=.5, fmt='.2f', vmin=-2.5, vmax=np.log10(1), cbar_kws={"shrink": .7})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.01, 0.05, 0.25, 1]))
    cbar.set_ticklabels([0.01, 0.05, 0.25, 1])

    # ax.set(xlabel='', ylabel='', title='Permutation p-values')
    ax.set_yticklabels(BRAIN_REGIONS, va='center', rotation=0)
    ax.set_xticklabels(test_names, rotation=90, ha='center')  # rotation=30, ha='right')
    #ax.set_title('Task-driven activity: Comparison across labs', loc='left', pad=15)

    return p_vals


def plot_panel_power_analysis(ax, ax2):

    significant_disturbances = pickle.load(open("new_max_metric.p", 'rb'))
    # max_y, min_y = 9, -3
    max_y, min_y = 14, -8

    obs_max, obs_min = -10, 10
    pad = 5 # in points
    i = -1
    perturbation_shift = 0.33
    dist_between_violins = 0.8
    lab_to_num = dict(zip(['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA'], list(np.arange(9) * dist_between_violins)))
    visualisation_plot = 'UCL'

    p_values = pickle.load(open("p_values_new_max_metric", 'rb'))
    df = load_dataframe()
    df_filt = filter_recordings(df, recompute=True, min_lab_region=2, min_rec_lab=0, min_neuron_region=2, freeze='release_2022_11')
    df_filt = df_filt[df_filt['permute_include'] == 1]
    df_filt_reg = df_filt.groupby('region')

    for jj, test in enumerate(tests.keys()):
        if test != 'post_stim':
            continue
        for ii, reg in enumerate(BRAIN_REGIONS):
            if reg != 'CA1':
                continue
            df_reg = df_filt_reg.get_group(reg)
            i += 1

            if test == 'avg_ff_post_move':
                data = df_reg[test].values
            else:
                data = df_reg['mean_fr_diff_{}'.format(test)].values
            labs = df_reg['institute'].values
            subjects = df_reg['subject'].values

            labs = labs[~np.isnan(data)]
            subjects = subjects[~np.isnan(data)]
            data = data[~np.isnan(data)]

            if significant_disturbances[i, 0, 0] == 0 and significant_disturbances[i, 0, 1] == 0:
                ax.plot([-0.3, 5.3], [min_y, max_y], 'k')
                ax.plot([-0.3, 5.3], [max_y, min_y], 'k')

            for j, lab in enumerate(np.unique(labs)):
                if np.sum(labs == lab) == 0:
                    continue
                if lab == 'UW':
                    continue

                lab_mean = data[labs == lab].mean()
                ax.plot([lab_to_num[lab] - 0.3, lab_to_num[lab] + 0.3], [lab_mean, lab_mean], color=lab_colors[lab])
                parts = ax.violinplot(data[labs == lab], positions=[lab_to_num[lab]], showextrema=False)
                parts['bodies'][0].set_facecolor(lab_colors[lab])
                parts['bodies'][0].set_edgecolor(lab_colors[lab])

                if lab == visualisation_plot:
                    parts = ax2.violinplot(data[labs == lab] + 1, positions=[0], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] - 1, positions=[0], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] + 2, positions=[0], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('red')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] - 2, positions=[0], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('red')
                    parts['bodies'][0].set_linestyles('--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 1, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 1, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 2, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 2, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 3, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 3, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 4, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('red')
                    parts['bodies'][0].set_linestyles('--')
                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 4, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    parts['bodies'][0].set_zorder(-1)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('red')
                    parts['bodies'][0].set_linestyles('--')

                    parts = ax2.violinplot(data[labs == lab], positions=[0], showextrema=False)
                    parts['bodies'][0].set_facecolor('grey')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts = ax2.violinplot(data[labs == lab], positions=[0], showextrema=False)
                    ax2.plot([0 - 0.3, 0 + 0.3], [np.mean(data[labs == lab]), np.mean(data[labs == lab])], color='grey', zorder=4)
                    parts['bodies'][0].set_zorder(0)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('white')
                    parts['bodies'][0].set_alpha(1)
                    parts = ax2.violinplot(data[labs == lab] * 1.4, positions=[dist_between_violins], showextrema=False)
                    ax2.plot([dist_between_violins - 0.3, dist_between_violins + 0.3], [np.mean(data[labs == lab]), np.mean(data[labs == lab])], color='grey', zorder=4)
                    parts['bodies'][0].set_facecolor('grey')
                    parts['bodies'][0].set_edgecolor('grey')
                    parts = ax2.violinplot(data[labs == lab] * 1.4, positions=[dist_between_violins], showextrema=False)
                    parts['bodies'][0].set_zorder(0)
                    parts['bodies'][0].set_facecolor('white')
                    parts['bodies'][0].set_edgecolor('white')
                    parts['bodies'][0].set_alpha(1)
                # parts['cmeans'].set_color('k') # this can be used to check whether the means align -> whether the datasets are assigned correctly

                val = significant_disturbances[i, j, 0]
                temp_color = lab_colors[lab] if val < 1000 else 'red'
                if temp_color == 'red':
                    val = max_y - lab_mean
                    print(ii + jj * 8 + 1)
                ax.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                if lab == visualisation_plot:
                    ax2.plot([0 + perturbation_shift, 0 + perturbation_shift], [lab_mean, lab_mean + 1], color='grey')
                    ax2.plot([dist_between_violins + perturbation_shift, dist_between_violins + perturbation_shift], [lab_mean, lab_mean + 3], color='grey')

                ax.axhline(0, color='grey', alpha=1/3, zorder=0)
                obs_max = max(obs_max, lab_mean + val)
                val = significant_disturbances[i, j, 1]
                temp_color = lab_colors[lab] if val > -1000 else 'red'
                if temp_color == 'red':
                    val = min_y - lab_mean
                    print(ii + jj * 8 + 1)
                ax.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                if lab == visualisation_plot:
                    ax2.plot([0 + perturbation_shift, 0 + perturbation_shift], [lab_mean, lab_mean - 1], color='grey')
                    ax2.plot([dist_between_violins + perturbation_shift, dist_between_violins + perturbation_shift], [lab_mean, lab_mean - 3], color='grey')

                obs_min = min(obs_min, lab_mean + val)
            ax.set_xlim(-0.3, 8 * dist_between_violins + .36)
            ax2.set_xlim(-0.3, dist_between_violins + .36)
            sns.despine()

            ax.set_ylim(min_y, max_y)
            ax2.set_ylim(min_y, max_y)

            ax2.set_ylabel('FR modulation (sp/s)')
            # ax.annotate("{}, {}, p={:.3f}".format(shortened_tests[test], reg, p_values[i]), xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
            ax2.set_xticks([])
            ax.set_xticks([])
            ax.set_yticks([])



def plot_power_analysis():

    significant_disturbances = pickle.load(open("new_max_metric.p", 'rb'))

    # max_y, min_y = 9, -3
    max_y, min_y = 16, -16

    obs_max, obs_min = -10, 10
    pad = 5 # in points
    i = -1
    perturbation_shift = 0.33
    dist_between_violins = 0.8
    lab_to_num = dict(zip(['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA'], list(np.arange(9) * dist_between_violins)))

    p_values = pickle.load(open("p_values_new_max_metric", 'rb'))
    df = load_dataframe()
    df_filt = filter_recordings(df, recompute=True, min_lab_region=2, min_rec_lab=0, min_neuron_region=2, freeze='release_2022_11')
    df_filt = df_filt[df_filt['permute_include'] == 1]
    df_filt_reg = df_filt.groupby('region')

    powers = []
    vars = []
    powers_ff = []
    vars_ff = []
    perturb_in_std = []
    all_powers = []
    ns = []

    fig = plt.figure(figsize=(1 * 12, 1.4142 * 12))
    for jj, test in enumerate(tests.keys()):
        for ii, reg in enumerate(BRAIN_REGIONS):

            plt.subplot(9, 5, ii + jj * 5 + 1)
            df_reg = df_filt_reg.get_group(reg)
            i += 1

            if test == 'avg_ff_post_move':
                data = df_reg[test].values
            else:
                data = df_reg['mean_fr_diff_{}'.format(test)].values
            labs = df_reg['institute'].values
            subjects = df_reg['subject'].values

            labs = labs[~np.isnan(data)]
            subjects = subjects[~np.isnan(data)]
            data = data[~np.isnan(data)]

            plt.title("p={:.3f}".format(p_values[i]))
            if significant_disturbances[i, 0, 0] == 0 and significant_disturbances[i, 0, 1] == 0:
                plt.plot([-0.3, 5.3], [min_y, max_y], 'k')
                plt.plot([-0.3, 5.3], [max_y, min_y], 'k')

            for j, lab in enumerate(np.unique(labs)):
                if np.sum(labs == lab) == 0:
                    continue
                if lab == 'UW':
                    continue

                lab_mean = data[labs == lab].mean()
                plt.plot([lab_to_num[lab] - 0.3, lab_to_num[lab] + 0.3], [lab_mean, lab_mean], color=lab_colors[lab])

                parts = plt.gca().violinplot(data[labs == lab], positions=[lab_to_num[lab]], showextrema=False)  # , showmeans=True)
                # print("{}, {}, {}".format(lab, lab_to_num[lab], np.min(data[labs == lab])))
                parts['bodies'][0].set_facecolor(lab_colors[lab])
                parts['bodies'][0].set_edgecolor(lab_colors[lab])

                if test == 'avg_ff_post_move':
                    vars_ff.append(np.std(data[labs == lab]) / np.sqrt(np.sum(labs == lab)))
                else:
                    vars.append(np.std(data[labs == lab]) / np.sqrt(np.sum(labs == lab)))

                # parts['cmeans'].set_color('k') # this can be used to check whether the means align -> whether the datasets are assigned correctly

                val = significant_disturbances[i, j, 0]

                perturb_in_std.append(val / np.std(data[labs == lab]))

                all_powers.append(val)
                ns.append(np.sum(labs == lab))
                if test == 'avg_ff_post_move':
                    powers_ff.append(val)
                    plt.axhline(1, color='grey', alpha=1/3, zorder=0)
                else:
                    powers.append(val)
                    plt.axhline(0, color='grey', alpha=1/3, zorder=0)

                temp_color = lab_colors[lab] if val < 1000 else 'red'
                if temp_color == 'red':
                    val = max_y - lab_mean
                    print(ii + jj * 8 + 1)
                plt.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                obs_max = max(obs_max, lab_mean + val)
                val = significant_disturbances[i, j, 1]

                all_powers[-1] -= val

                if test == 'avg_ff_post_move':
                    powers_ff[-1] -= val
                else:
                    powers[-1] -= val

                perturb_in_std.append(val / np.std(data[labs == lab]))

                temp_color = lab_colors[lab] if val > -1000 else 'red'
                if temp_color == 'red':
                    val = min_y - lab_mean
                    print(ii + jj * 8 + 1)
                plt.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                obs_min = min(obs_min, lab_mean + val)
            plt.xlim(-0.3, 8 * dist_between_violins + .36)
            sns.despine()
            if jj == 7:
                if ii == 0:
                    plt.ylabel('Fano factor', fontsize=18)
                plt.ylim(0, 4)
            else:
                plt.ylim(min_y, max_y)
            if ii != 0 and ii != 7:
                plt.gca().set_yticks([])
            if ii == 0:
                plt.gca().annotate(shortened_tests[test], xy=(-0.45, 0.5), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', size='x-large', ha='right', va='center', rotation='vertical')
                if jj == 3:
                    plt.ylabel('FR modulation (sp/s)', fontsize=21)
            if jj == 0:
                plt.gca().annotate(reg, xy=(0.5, 1.2), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='x-large', ha='center', va='baseline')
            if jj != 7:
                plt.gca().set_xticks([])
            else:
                plt.gca().set_xticks([])
                plt.xlabel('Labs', fontsize=18)

    plt.subplot(9, 5, ii + jj * 5 + 1 + 2)
    plt.scatter(powers, np.power(vars, 0.5), color='blue', label="Firing modulation", s=0.15)
    plt.scatter(powers_ff, np.power(vars_ff, 0.5), color='blue', label="Fano factor", s=0.15)
    # plt.xlabel("Shifts", size=14)
    # plt.ylabel("Std / sqrt(N)", size=14)
    plt.gca().annotate("Shifts", xy=(0.5, -0.45), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    plt.gca().annotate(r"Std / $\sqrt{N}$", xy=(-0.25, 0.2), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline', rotation='vertical')
    # plt.legend(fontsize=14, frameon=False)
    plt.xlim(0)
    plt.ylim(0)
    sns.despine()

    plt.subplot(9, 5, ii + jj * 5 + 1 + 4)
    plt.hist(perturb_in_std, bins=25, color='grey')
    # plt.xlabel("Shifts (std)", size=14)
    # plt.ylabel("# of occurences", size=14)
    plt.gca().annotate("Shifts (std)", xy=(0.5, -0.45), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    plt.gca().annotate("# occurences", xy=(-0.25, 0.0), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline', rotation='vertical')
    plt.legend(fontsize=14, frameon=False)
    plt.ylim(0)
    sns.despine()

    fig.subplots_adjust(hspace=0.27)
    fig.subplots_adjust(wspace=0.05)
    # plt.tight_layout()
    # fig.subplots_adjust(left=0.12)
    plt.savefig(fig_path.joinpath('fig_power_analysis.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('fig_power_analysis.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.scatter(powers, np.power(vars, 0.5), color='blue', label="Firing modulation")
    plt.scatter(powers_ff, np.power(vars_ff, 0.5), color='red', label="Fano factor")
    plt.xlabel("Range of possible shift", size=14)
    plt.ylabel("Standard deviation / sqrt(N)", size=14)
    plt.legend(fontsize=14, frameon=False)
    plt.xlim(0, 4.5)
    plt.ylim(0, 1.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig("limited scattering")
    plt.show()

    from scipy.stats import pearsonr
    print(pearsonr(powers, vars))
    print(pearsonr(powers_ff, vars_ff))

    print(obs_min, obs_max)
    return powers, vars, powers_ff, vars_ff, perturb_in_std, all_powers, ns


def power_analysis_to_table():
    power_an = pickle.load(open("new_max_metric.p", 'rb'))
    local_labs = ['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA']
    lab_to_num = dict(zip(local_labs, range(len(local_labs))))

    df = load_dataframe()
    df_filt = filter_recordings(df, recompute=True, min_lab_region=2, min_rec_lab=0, min_neuron_region=2, freeze='release_2022_11')
    df_filt = df_filt[df_filt['permute_include'] == 1]
    df_filt_reg = df_filt.groupby('region')

    print("& {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline".format(*['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA']))
    inside_string = " & {}, {}"
    formatting_string = "{} {}" + len(local_labs) * inside_string + " \\\\ \\hline"
    i = -1  # probably
    for test in tests.keys():
        test_name = shortened_tests[test]
        for reg in BRAIN_REGIONS:
            df_reg = df_filt_reg.get_group(reg)
            i += 1

            labs = df_reg['institute'].values
            vals = ['-'] * len(local_labs) * 2

            for j, lab in enumerate(np.unique(labs)):
                if lab == 'UW':
                    continue
                val = power_an[i, j, 0]
                vals[lab_to_num[lab] * 2] = "$\\infty$" if val > 1000 else np.round(val, 2)
                val = power_an[i, j, 1]
                vals[lab_to_num[lab] * 2 + 1] = "$-\\infty$" if val < -1000 else np.round(val, 2)

            print(formatting_string.format(test_name, reg, *vals))


def find_sig_p_value(p_values_to_copy, i):
    # take an array of p_values, and index i specifying the relevant one
    # see what p[i] what need to be, to be significant after correction
    p_values = p_values_to_copy.copy()
    _, corrected_p_vals, _, _ = multipletests(p_values, 0.05, method='fdr_bh')
    if corrected_p_vals[i] < 0.05:
        return p_values[i], corrected_p_vals[i]
    actual_p = p_values[i]
    p_attempt = np.round(actual_p / 2, 5)
    step_unit = p_attempt
    j = 0
    while True:
        j += 1
        p_values[i] = np.round(p_attempt, 5)
        _, corrected_p_vals_low, _, _ = multipletests(p_values, 0.05, method='fdr_bh')
        p_values[i] += 0.00002
        _, corrected_p_vals_high, _, _ = multipletests(p_values, 0.05, method='fdr_bh')
        if corrected_p_vals_low[i] < 0.05 and corrected_p_vals_high[i] >= 0.05:
            return p_values[i] - 0.00002, corrected_p_vals_low[i]
        if j == 100:
            print("Failed, current p_value is {}".format(corrected_p_vals_low[i]))
            return p_values[i] - 0.00002, corrected_p_vals_low[i]
        if corrected_p_vals_low[i] >= 0.05:
            p_attempt -= step_unit * 0.5 ** j
        else:
            p_attempt += step_unit * 0.5 ** j

def find_sig_manipulation(data, lab_to_manip, labs, subjects, p_to_reach, direction='positive', sensitivity=0.01):
    lower_bound = 0 if direction == 'positive' else -1000
    higher_bound = 1000 if direction == 'positive' else 0

    found_bound = False
    bound = 0
    while not found_bound:
        bound += 10 if direction == 'positive' else -10
        p = permut_test(data + (labs == lab_to_manip) * bound, metric=distribution_dist_approx_max, labels1=labs,
                        labels2=subjects, shuffling='labels1_based_on_2', n_cores=8, n_permut=20000)
        if p < p_to_reach:
            found_bound = True
            if direction == 'positive':
                higher_bound = bound
            else:
                lower_bound = bound
        else:
            if direction == 'positive':
                lower_bound = bound
            else:
                higher_bound = bound
        if not found_bound:
            if direction == 'positive':
                if bound > data.max() + 20:
                    print("Failed to find")
                    return -np.inf, np.inf
            else:
                if bound < data.min() - 20:
                    print("Failed to find")
                    return -np.inf, np.inf

    while np.abs(lower_bound - higher_bound) > sensitivity:

        test = (lower_bound + higher_bound) / 2
        p = permut_test(data + (labs == lab_to_manip) * test, metric=distribution_dist_approx_max, labels1=labs,
                        labels2=subjects, shuffling='labels1_based_on_2', n_cores=8, n_permut=20000)
        if p < p_to_reach:
            if direction == 'positive':
                higher_bound = test
            else:
                lower_bound = test
        else:
            if direction == 'positive':
                lower_bound = test
            else:
                higher_bound = test
    return lower_bound, higher_bound


# qcs = ["pass", "high_noise", "low_yield", "missed_target", "artifacts", "low_trials", "high_lfp", "all"]
# for qc in qcs:
#     print(qc)
#     plot_panel_permutation(qc=qc, n_permut=90000, n_cores=10)
#     plt.title(qc)
#     plt.savefig("temp_{}".format(qc))
#     plt.close()

import pickle

plot_main_figure()
# power_analysis_to_table()
# quit()
a = plot_power_analysis()
quit()

significant_disturbances = pickle.load(open("new_max_metric.p", 'rb'))
plot_power_analysis(significant_disturbances)
quit()

plot_panel_permutation()
p_values = pickle.load(open("p_values_new_max_metric", 'rb'))  # renew by calling plot_panel_permutation
print(p_values)
print(np.sum(p_values < 0.01))


df = load_dataframe()
df_filt = filter_recordings(df, recompute=True, min_lab_region=2, min_rec_lab=0, min_neuron_region=2, freeze='release_2022_11')
df_filt = df_filt[df_filt['permute_include'] == 1]

df_filt_reg = df_filt.groupby('region')
results = pd.DataFrame()


n_power_simulations = 1000
p_values_from_null = np.zeros((5, n_power_simulations))
power_ps = []
i = -1
significant_disturbances = np.zeros((len(p_values), 9, 2))
for test in tests.keys():
    print(test)
    for jj, reg in enumerate(BRAIN_REGIONS):
        i += 1

        df_reg = df_filt_reg.get_group(reg)

        if test == 'avg_ff_post_move':
            data = df_reg[test].values
        else:
            data = df_reg['mean_fr_diff_{}'.format(test)].values
        labs = df_reg['institute'].values
        subjects = df_reg['subject'].values

        labs = labs[~np.isnan(data)]
        subjects = subjects[~np.isnan(data)]
        data = data[~np.isnan(data)]

        if p_values[i] < 0.01:
            print("test already significant {}, {}, {}".format(test, reg, i))
            # p = permut_test(data, metric=distribution_dist_approx_max, labels1=labs, title="{} {}".format(test[:4], reg),
            #                 labels2=subjects, shuffling='labels1_based_on_2', n_permut=10)
            continue

        # continue

        #power_ps.append(pickle.load(open("ps_from_null_{}.p".format(test), 'rb')))
        #continue

        # print("power analysis")
        # new_labs, new_subjects = shuffle_labels(labs, subjects, n_power_simulations, shuffling='labels1_based_on_2')
        #
        # for j in range(n_power_simulations):
        #     a = time.time()
        #     p = permut_test(data, metric=distribution_dist_approx_max, labels1=new_labs[j],
        #                     labels2=new_subjects[j], shuffling='labels1_based_on_2', n_permut=10000)
        #     p_values_from_null[jj, j] = p
        #     print(time.time() - a)
        #     quit()

        # print("shuffle test")
        # permuted_labels1, permuted_labels2 = shuffle_labels(labels1=labs, labels2=subjects, n_permut=100000, n_cores=5, shuffling='labels1_based_on_2')

        # terrible_counter = 0
        # iiis = []
        # for iii in range(100000):
        #     for local_lab in range(6):
        #         if set(np.unique(permuted_labels2[0][permuted_labels1[iii] == local_lab])) <= set([0, 1, 2, 3]):
        #             terrible_counter += 1
        #             iiis.append(iii)

        # print(terrible_counter)

        # quit()

        # print('normal test')
        # p = permut_test(data, metric=distribution_dist_approx_max, labels1=labs,
        #                 labels2=subjects, shuffling='labels1_based_on_2', n_cores=3, n_permut=30000)
        #
        # quit()

        # for j, manipulate_lab in enumerate(np.unique(labs)):
        #     if significant_disturbances[i, j, 0] > 100:
        #         print('pos')
        #     if significant_disturbances[i, j, 1] < -100:
        #         print('neg')
        #         quit()
        # continue

        print(np.unique(labs))

        for j, manipulate_lab in enumerate(np.unique(labs)):
            # if significant_disturbances[i, j, 0] < 100 and significant_disturbances[i, j, 1] > -100:
            #     continue
            lower, higher = find_sig_manipulation(data.copy(), manipulate_lab, labs, subjects, 0.01, 'positive')
            significant_disturbances[i, j, 0] = higher
            print("found bound: {}".format(higher))
            lower, higher = find_sig_manipulation(data.copy(), manipulate_lab, labs, subjects, 0.01, 'negative')
            significant_disturbances[i, j, 1] = lower
            print("found bound: {}".format(lower))
        pickle.dump(significant_disturbances, open("new_max_metric.p", 'wb'))
# pickle.dump(p_values_from_null, open("ps_from_null_{}.p".format(test), 'wb'))
quit()


power_analysis_to_table()


quit()

if __name__ == '__main__':
    plot_main_figure()
