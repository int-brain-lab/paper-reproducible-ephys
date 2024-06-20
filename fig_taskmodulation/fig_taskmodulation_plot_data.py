import matplotlib.pyplot as plt
import numpy as np
import figrid as fg
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs, save_figure_path, figure_style, save_data_path
from fig_taskmodulation.fig_taskmodulation_load_data import load_data, load_dataframe, tests, filtering_criteria
from fig_taskmodulation.fig_taskmodulation_plot_functions import plot_raster_and_psth, plot_raster_and_psth_LvsR
import seaborn as sns
import pandas as pd
import pickle
from matplotlib.transforms import Bbox
import json
from permutation_test import permut_test, distribution_dist_approx_max, permut_dist
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
from scipy.stats import sem


np.set_printoptions(suppress=True)

lab_number_map, institution_map, lab_colors = labs()
fig_path = save_figure_path(figure='fig_taskmodulation')

PRINT_PIDS = False

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


shortened_tests = {#'trial': 'Trial (first 400 ms)',
                   'post_stim': 'Stimulus',
                   'post_move': 'Movement',
                   'start_to_move': 'Reaction period',
                   'pre_move': 'Move. initiation',
                   'post_reward': 'Reward',
                   'pre_move_lr': 'L vs R move.',
                   'avg_ff_post_move': 'Fano Factor'}

region_rename = dict(zip(BRAIN_REGIONS, ['VISa/am', 'CA1', 'DG', 'LP', 'PO']))

def plot_main_figure():
    DPI = 400  # if the figure is too big on your screen, lower this number
    figure_style()
    fig = plt.figure(figsize=(7, 10.5), dpi=500)  # full width figure is 7 inches
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
    ax = {'panel_A_1': fg.place_axes_on_grid(fig, xspan=[0.08, 0.288], yspan=[0.045, 0.1],
                                             wspace=0.3),
          'panel_A_2': fg.place_axes_on_grid(fig, xspan=[0.08, 0.288], yspan=[0.13, 0.2],
                                             wspace=0.3),
          'panel_B': fg.place_axes_on_grid(fig, xspan=[0.388, 0.631], yspan=[0.045, 0.2],
                                           wspace=0.3),
          'panel_C': fg.place_axes_on_grid(fig, xspan=[0.741, 1.], yspan=[0.045, 0.2],
                                           wspace=0.3),
          'panel_D_1': fg.place_axes_on_grid(fig, xspan=[0.075,  0.27375], yspan=[0.29, 0.44],
                                             wspace=0.3),
          'panel_D_2': fg.place_axes_on_grid(fig, xspan=[0.28375, 0.4825], yspan=[0.29, 0.44],
                                             wspace=0.3),
          'panel_D_3': fg.place_axes_on_grid(fig, xspan=[0.4925, 0.69125], yspan=[0.29, 0.44],
                                             wspace=0.3),
          'panel_D_4': fg.place_axes_on_grid(fig, xspan=[0.70125, .9], yspan=[0.29, 0.44],
                                             wspace=0.3),
          'panel_E_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.55], yspan=[0.6, 0.68],
                                             wspace=0.3),
          'panel_E_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.55], yspan=[0.69, 0.76],
                                             wspace=0.3),
          'panel_E_3': fg.place_axes_on_grid(fig, xspan=[0.075, 0.55], yspan=[0.77, 0.84],
                                             wspace=0.3),
          'panel_E_4': fg.place_axes_on_grid(fig, xspan=[0.075, 0.55], yspan=[0.85, .92],
                                             wspace=0.3),
          'panel_E_5': fg.place_axes_on_grid(fig, xspan=[0.075, 0.55], yspan=[0.93, 1.],
                                             wspace=0.3),
          'panel_F_1': fg.place_axes_on_grid(fig, xspan=[0.64, 0.71], yspan=[0.51, .69],
                                             wspace=0.3),
          'panel_F_3': fg.place_axes_on_grid(fig, xspan=[0.73, 1.], yspan=[0.51, .69],
                                             wspace=0.3),
          'panel_F_2': fg.place_axes_on_grid(fig, xspan=[0.64, 1.], yspan=[0.735, .98],
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
    df_filt = filter_recordings(df, **filtering_criteria)
    df_filt = df_filt[df_filt['permute_include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby('region')
    max_neurons = 0
    min_neurons = 1000000
    for iR, reg in enumerate(BRAIN_REGIONS):
        df_reg = df_filt_reg.get_group(reg)
        df_reg_inst = df_reg.groupby('institute')
        for inst in df_reg_inst.groups.keys():
            inst_idx = df_reg_inst.groups[inst]
            max_neurons = max(max_neurons, inst_idx.shape[0])
            min_neurons = min(min_neurons, inst_idx.shape[0])
    D_regions = [reg for reg in BRAIN_REGIONS if reg != 'LP']
    plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons, ax=[ax['panel_C']], save=False, plotted_regions=['LP'])
    plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons, ax=[ax['panel_D_1'], ax['panel_D_2'], ax['panel_D_3'], ax['panel_D_4']], save=False, plotted_regions=D_regions)

    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0.007, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'b', 'xpos': 0.305, 'ypos': 0.007, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'c', 'xpos': 0.66, 'ypos': 0.007, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'd', 'xpos': 0, 'ypos': 0.26, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'e', 'xpos': 0, 'ypos': 0.51, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'f', 'xpos': 0.59, 'ypos': 0.51, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'g', 'xpos': 0.708, 'ypos': 0.51, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'h', 'xpos': 0.59, 'ypos': 0.69, 'fontsize': 10, 'weight': 'bold'}]

    fg.add_labels(fig, labels)
    print(f'Saving figures to {fig_path}')
    plt.savefig(fig_path.joinpath('fig_taskmodulation_combined.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('fig_taskmodulation_combined.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()




def task_mod_panel_helper(fig, panel_name, xspan, ybot, ytop):
    step = (ytop - ybot + 0.01) / 5
    ret = {panel_name + '1': fg.place_axes_on_grid(fig, xspan=xspan, yspan=[ybot + 0 * step, ybot + 1 * step - 0.01],
                                                   wspace=0.3),
           panel_name + '2': fg.place_axes_on_grid(fig, xspan=xspan, yspan=[ybot + 1 * step, ybot + 2 * step - 0.01],
                                                   wspace=0.3),
           panel_name + '3': fg.place_axes_on_grid(fig, xspan=xspan, yspan=[ybot + 2 * step, ybot + 3 * step - 0.01],
                                                   wspace=0.3),
           panel_name + '4': fg.place_axes_on_grid(fig, xspan=xspan, yspan=[ybot + 3 * step, ybot + 4 * step - 0.01],
                                                   wspace=0.3),
           panel_name + '5': fg.place_axes_on_grid(fig, xspan=xspan, yspan=[ybot + 4 * step, ytop],
                                                   wspace=0.3)}
    return ret


def plot_supp_figure():
    DPI = 400  # if the figure is too big on your screen, lower this number
    figure_style()
    fig = plt.figure(figsize=(7, 10.5), dpi=500)  # full width figure is 7 inches
    panel_a = task_mod_panel_helper(fig, 'panel_A_', [0.075, 0.45], 0.11, 0.32)
    panel_b = task_mod_panel_helper(fig, 'panel_B_', [0.55, 1.], 0.11, 0.32)
    panel_c = task_mod_panel_helper(fig, 'panel_C_', [0.075, 0.45], 0.44, 0.65)
    panel_d = task_mod_panel_helper(fig, 'panel_D_', [0.55, 1.], 0.44, 0.65)
    panel_e = task_mod_panel_helper(fig, 'panel_E_', [0.075, 0.45], 0.77, 1)
    panel_f = task_mod_panel_helper(fig, 'panel_F_', [0.55, 1.], 0.77, 1)

    plot_panel_task_modulated_neurons(specific_tests=['post_stim'],
                                      ax=[panel_a['panel_A_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['post_move'],
                                      ax=[panel_b['panel_B_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['start_to_move'],
                                      ax=[panel_c['panel_C_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['pre_move'],
                                      ax=[panel_d['panel_D_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['post_reward'],
                                      ax=[panel_e['panel_E_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['pre_move_lr'],
                                      ax=[panel_f['panel_F_{}'.format(x)] for x in range(1, 6)],
                                      save=False)

    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'b', 'xpos': 0.5, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'c', 'xpos': 0, 'ypos': 0.33, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'd', 'xpos': 0.5, 'ypos': 0.33, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'e', 'xpos': 0, 'ypos': 0.66, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'f', 'xpos': 0.5, 'ypos': 0.66, 'fontsize': 10, 'weight': 'bold'}]
    fg.add_labels(fig, labels)

    print(f'Saving figures to {fig_path}')
    plt.savefig(fig_path.joinpath('figure_taskmodulation_supp.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('figure_taskmodulation_supp.pdf'), bbox_inches='tight', pad_inches=0)
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


def plot_panel_single_neuron_LvsR(ax=None, save=True, neuron=650):
    # Does not distinguish between contrasts, but distinguishes by side

    pid = 'f26a6ab1-7e37-4f8d-bb50-295c056e1062'
    align_event = 'move'
    params = {'smoothing': 'sliding',
              'fr_bin_size': 0.06,
              'event_epoch': [-0.2, 0.2],  # [-0.3, 0.22],
              'slide_kwargs_fr': {'n_win': 3, 'causal': 1}}

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

    df_filt = filter_recordings(df, **filtering_criteria)
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

    df_filt = filter_recordings(df, **filtering_criteria)
    all_frs_l = data['all_frs_l'][df_filt['permute_include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['permute_include'] == 1]
    all_frs_l_std = data['all_frs_l_std'][df_filt['permute_include'] == 1]
    all_frs_r_std = data['all_frs_r_std'][df_filt['permute_include'] == 1]
    df_filt = df_filt[df_filt['permute_include'] == 1].reset_index()

    if PRINT_PIDS:
        json.dump(list(np.unique(df_filt['pid'])), open("panel d", 'w'))
        return

    # Example to get similar plot to figure 4c
    if ax is None:
        fig, ax = plt.subplots(1, len(plotted_regions))
    df_filt_reg = df_filt.groupby('region')

    min_lw = 0.25
    max_lw = 2.5

    print("max neurons: {}; min neurons: {}".format(max_neurons, min_neurons))

    from scipy.stats import sem
    all_present_labs = []
    for iR, reg in enumerate(plotted_regions):
        df_reg = df_filt_reg.get_group(reg)
        df_reg_inst = df_reg.groupby('institute')
        for inst in df_reg_inst.groups.keys():
            df_inst = df_reg_inst.get_group(inst)
            inst_idx = df_reg_inst.groups[inst]

            # Select L vs R side:
            # frs_inst = all_frs_l[inst_idx, :]
            frs_inst = all_frs_r[inst_idx, :]

            if inst not in all_present_labs:
                all_present_labs.append(df_inst.iloc[0]['institute'])
            print(inst, reg, inst_idx.shape[0])
            ax[iR].errorbar(data['time'], np.mean(frs_inst, axis=0), yerr=sem(frs_inst, axis=0) / 2, c=lab_colors[inst],
                        lw=min_lw + ((inst_idx.shape[0] - min_neurons) / (max_neurons - min_neurons)) * max_lw,
                        alpha=0.8)
        ax[iR].set_ylim(bottom=-4, top=8)
        ax[iR].set_yticks([-4, 0, 4, 8])
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
        # ax[iR].set_title(region_rename[reg])

        if len(plotted_regions) == 1:
            ax[iR].set_title('Recording averages in LP', loc='left')
        else:
            ax[iR].set_title(region_rename[reg])

        if iR == 1 or len(plotted_regions) == 1:
            ax[iR].set_xlabel("Time from movement onset (s)")

        print(all_present_labs)
        if iR == len(plotted_regions) - 1 and len(plotted_regions) != 1:
            # this is a hack for the legend
            for lab in all_present_labs:
                print(lab, lab_colors[lab])
                ax[iR].plot(data['time'], np.zeros_like(data['time']) - 100, c=lab_colors[lab], label=lab)
                leg = ax[iR].legend(frameon=False, bbox_to_anchor=(1, 1.19), labelcolor='linecolor', handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)

    if save:
        plt.savefig(fig_path.joinpath('fig_taskmodulation_all_subjects.png'))


def plot_panel_task_modulated_neurons(specific_tests=None, ax=None, save=True):

    # load dataframe from prev fig. 5 (To be combined with new Fig 4)
    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria)
    df_filt = df_filt[df_filt['permute_include'] == 1]

    if PRINT_PIDS:
        json.dump(list(np.unique(df_filt['pid'])), open("panel e", 'w'))
        return

    # Group data frame by region
    df_region = df_filt.groupby('region')
    lab_to_num = dict(zip(['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA', 'UW'], list(range(10))))

    names = tests.keys() if specific_tests is None else specific_tests
    # FIGURE 5c and supplementary figures
    for test in names:
        for i, br in enumerate(BRAIN_REGIONS):
            df_br = df_region.get_group(br)

            df_inst = df_br.groupby(['subject', 'institute'], as_index=False)
            vals = df_inst[test].mean().sort_values('institute')
            colors = [lab_colors[col] for col in vals['institute'].values]
            positions = [lab_to_num[lab] + np.random.rand() * 0.1 - 0.05 for lab in vals['institute'].values]

            if ax is None:
                plt.subplot(len(BRAIN_REGIONS), 1, i + 1)
                plt.bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
                plt.ylim(bottom=0, top=1)
                plt.ylabel(region_rename[br])
                plt.yticks([0, 1], [0, 1])
                plt.xticks([])
                sns.despine()
                if i == 4:
                    plt.xlabel('Mice')
                elif i == 0:
                    plt.title('% modulated neurons', loc='left')
            else:
                for lab in vals['institute'].values:
                    mean = vals[test].values[vals['institute'].values == lab].mean()
                    stan_err = sem(vals[test].values[vals['institute'].values == lab])
                    ax[i].plot([mean, mean], [lab_to_num[lab]-0.2, lab_to_num[lab]+0.2], color=lab_colors[lab], lw=1)
                    ax[i].plot([mean-stan_err / 2, mean+stan_err / 2], [lab_to_num[lab], lab_to_num[lab]], lw=0.5, color=lab_colors[lab])
                ax[i].scatter(vals[test].values, positions, color=colors, s=1)
                # ax[i].bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
                ax[i].set_xlim(-0.05, 1.05)
                ax[i].set_ylabel(region_rename[br], labelpad=0)
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                sns.despine()
                
                # if i == 0:
                #     ax[i].set_title('{} test'.format(shortened_tests[test]))
                if i == 4:
                    ax[i].set_xticks([0, 1], [0, 100])  # we are plotting percentages
                    ax[i].set_xlabel('% modulated neurons ({} test)'.format(shortened_tests[test]))
        if specific_tests is None:
            plt.suptitle(tests[test], size=22)
        if save:
            plt.savefig(fig_path.joinpath(test))


def plot_panel_permutation(ax=None, n_permut=20000, qc='pass', n_cores=8):
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
    test_names = [shortened_tests[test] for test in tests.keys()]

    shape = (len(tests.keys()), len(BRAIN_REGIONS))
    p_vals_perc_mod = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('p_values_percent_modulated'), 'rb'))
    p_vals = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('p_values'), 'rb'))
    # _, corrected_p_vals, _, _ = multipletests(results.p_value_permut.values, 0.05, method='fdr_bh')
    p_vals = p_vals.reshape(shape)
    p_vals_perc_mod = p_vals_perc_mod.reshape(shape)
    # corrected_p_vals = results.p_value_permut.values.reshape(shape)

    RdYlGn = cm.get_cmap('RdYlGn', 256)(np.linspace(0, 1, 800))
    color_array = np.vstack([np.tile(np.concatenate((to_rgb('darkviolet'), [1])), (200, 1)), RdYlGn, np.tile(np.concatenate((to_rgb('white'), [1])), (1, 1))])
    newcmp = ListedColormap(color_array)
    ax = sns.heatmap(np.vstack([np.clip(np.log10(p_vals_perc_mod.T), None, - 2.5/800), np.log10(np.ones(shape[0])), np.clip(np.log10(p_vals.T), None, - 2.5/800)]), cmap=newcmp, square=True,
                     cbar=True, annot=False, annot_kws={"size": 12}, ax=ax,
                     linewidths=.5, fmt='.2f', vmin=-2.5, vmax=np.log10(1), cbar_kws={"shrink": .7})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.01, 0.05, 0.25, 1]))
    cbar.set_ticklabels([0.01, 0.05, 0.25, 1])

    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    ax.set_yticklabels([region_rename[br] for br in BRAIN_REGIONS] + [region_rename[br] for br in BRAIN_REGIONS], va='center', rotation=0)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    ax.set_xticklabels(test_names, ha='center', rotation=90)  # rotation=30, ha='right')
    #ax.set_title('Task-driven activity: Comparison across labs', loc='left', pad=15)

    ax.set_title("% modulated neurons", size='small')
    ax.annotate("Distribution of F.R. modulations", (0.5, 0.485), horizontalalignment='center', xycoords='axes fraction', size='small')

    return p_vals


def plot_panel_power_analysis(ax, ax2):

    significant_disturbances = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('shifts'), 'rb'))
    # max_y, min_y = 9, -3
    max_y, min_y = 14, -8

    obs_max, obs_min = -10, 10
    i = -1
    perturbation_shift = 0.3
    dist_between_violins = 0.8
    lab_to_num = dict(zip(['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA', 'UW'], list(np.arange(10) * dist_between_violins)))
    visualisation_plot = 'CSHL (C)'

    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria)
    df_filt = df_filt[df_filt['permute_include'] == 1]
    df_filt_reg = df_filt.groupby('region')

    if PRINT_PIDS:
        json.dump(list(np.unique(df_filt['pid'])), open("panel g", 'w'))
        print("Overall num of PIDs: {}".format(len(np.unique(df_filt['pid']))))

        temp = df_filt[(df_filt['permute_include'] == 1) & (df_filt['region'] == BRAIN_REGIONS[0])]
        json.dump(list(np.unique(temp['pid'])), open("panel g in {}".format(BRAIN_REGIONS[0]), 'w'))
        print("{} has {} PIDs".format(BRAIN_REGIONS[0], len(np.unique(temp['pid']))))

        temp = df_filt[(df_filt['permute_include'] == 1) & (df_filt['region'] == BRAIN_REGIONS[1])]
        json.dump(list(np.unique(temp['pid'])), open("panel g in {}".format(BRAIN_REGIONS[1]), 'w'))
        print("{} has {} PIDs".format(BRAIN_REGIONS[1], len(np.unique(temp['pid']))))

        temp = df_filt[(df_filt['permute_include'] == 1) & (df_filt['region'] == BRAIN_REGIONS[2])]
        json.dump(list(np.unique(temp['pid'])), open("panel g in {}".format(BRAIN_REGIONS[2]), 'w'))
        print("{} has {} PIDs".format(BRAIN_REGIONS[2], len(np.unique(temp['pid']))))

        temp = df_filt[(df_filt['permute_include'] == 1) & (df_filt['region'] == BRAIN_REGIONS[3])]
        json.dump(list(np.unique(temp['pid'])), open("panel g in {}".format(BRAIN_REGIONS[3]), 'w'))
        print("{} has {} PIDs".format(BRAIN_REGIONS[3], len(np.unique(temp['pid']))))

        temp = df_filt[(df_filt['permute_include'] == 1) & (df_filt['region'] == BRAIN_REGIONS[4])]
        json.dump(list(np.unique(temp['pid'])), open("panel g in {}".format(BRAIN_REGIONS[4]), 'w'))
        print("{} has {} PIDs".format(BRAIN_REGIONS[4], len(np.unique(temp['pid']))))

        return

    for jj, test in enumerate(tests.keys()):
        if test != 'post_stim':
            i += 1 # this should probably be in the next loop and only works because the picked test is the first in the dict, TODO
            continue
        for ii, reg in enumerate(BRAIN_REGIONS):
            if reg != 'CA1':
                i += 1
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
                if isinstance(temp_color, str):
                    val = max_y - lab_mean
                    # print(ii + jj * 8 + 1)
                ax.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                if lab == visualisation_plot:
                    ax2.plot([0 + perturbation_shift, 0 + perturbation_shift], [lab_mean, lab_mean + 1], color='grey')
                    ax2.plot([dist_between_violins + perturbation_shift, dist_between_violins + perturbation_shift], [lab_mean, lab_mean + 3], color='grey')

                ax.axhline(0, color='grey', alpha=1/3, zorder=0)
                obs_max = max(obs_max, lab_mean + val)
                val = significant_disturbances[i, j, 1]
                temp_color = lab_colors[lab] if val > -1000 else 'red'
                if isinstance(temp_color, str):
                    val = min_y - lab_mean
                    # print(ii + jj * 8 + 1)
                ax.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                if lab == visualisation_plot:
                    ax2.plot([0 + perturbation_shift, 0 + perturbation_shift], [lab_mean, lab_mean - 1], color='grey')
                    ax2.plot([dist_between_violins + perturbation_shift, dist_between_violins + perturbation_shift], [lab_mean, lab_mean - 3], color='grey')

                obs_min = min(obs_min, lab_mean + val)
            ax.set_xlim(-0.3, (len(lab_to_num) - 1) * dist_between_violins + .36)
            sns.despine(ax=ax)
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

    significant_disturbances = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('shifts'), 'rb'))

    # max_y, min_y = 9, -3
    max_y, min_y = 16, -16
    ff_min_y, ff_max_y = 0, 4

    obs_max, obs_min = -10, 10
    pad = 5 # in points
    i = -1
    perturbation_shift = 0.33
    dist_between_violins = 0.8
    lab_to_num = dict(zip(['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA', 'UW'], list(np.arange(10) * dist_between_violins)))

    p_values = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('p_values'), 'rb'))
    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria, recompute=False)
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

            plt.subplot(8, 5, ii + jj * 5 + 1)
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
                if jj == 6:
                    plt.plot([-0.3, 6.3], [ff_min_y, ff_max_y], 'k')
                    plt.plot([-0.3, 6.3], [ff_max_y, ff_min_y], 'k')
                else:
                    plt.plot([-0.3, 6.3], [min_y, max_y], 'k')
                    plt.plot([-0.3, 6.3], [max_y, min_y], 'k')

            for j, lab in enumerate(np.unique(labs)):
                if np.sum(labs == lab) == 0:
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

                if not (significant_disturbances[i, 0, 0] == 0 and significant_disturbances[i, 0, 1] == 0):
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
                if isinstance(temp_color, str):
                    val = max_y - lab_mean
                    # print(ii + jj * 8 + 1)
                plt.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                obs_max = max(obs_max, lab_mean + val)
                val = significant_disturbances[i, j, 1]

                all_powers[-1] -= val

                if test == 'avg_ff_post_move':
                    powers_ff[-1] -= val
                else:
                    powers[-1] -= val

                if not (significant_disturbances[i, 0, 0] == 0 and significant_disturbances[i, 0, 1] == 0):
                    perturb_in_std.append(val / np.std(data[labs == lab]))

                temp_color = lab_colors[lab] if val > -1000 else 'red'
                if isinstance(temp_color, str):
                    val = min_y - lab_mean
                    # print(ii + jj * 8 + 1)
                plt.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                obs_min = min(obs_min, lab_mean + val)
            plt.xlim(-0.3, (len(lab_to_num) - 1) * dist_between_violins + .36)
            sns.despine()
            if jj == 6:
                if ii == 0:
                    plt.ylabel('Fano factor', fontsize=18)
                plt.ylim(ff_min_y, ff_max_y)
            else:
                plt.ylim(min_y, max_y)
            if ii != 0 and ii != 6:
                plt.gca().set_yticks([])
            if ii == 0:
                plt.gca().annotate(shortened_tests[test], xy=(-0.45, 0.5), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', size='x-large', ha='right', va='center', rotation='vertical')
                if jj == 3:
                    plt.ylabel('FR modulation (sp/s)', fontsize=21)
            if jj == 0:
                plt.gca().annotate(region_rename[reg], xy=(0.5, 1.2), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='x-large', ha='center', va='baseline')
            if jj != 6:
                plt.gca().set_xticks([])
            else:
                plt.gca().set_xticks([])
                plt.xlabel('Labs', fontsize=18)

    plt.subplot(8, 5, ii + jj * 5 + 1 + 2)
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

    std_limit = 3
    plt.subplot(8, 5, ii + jj * 5 + 1 + 4)
    perturb_in_std = np.array(perturb_in_std)
    assert np.sum(perturb_in_std == 0) == 0, "Perturbations of significant tests in histogram"
    plt.hist(perturb_in_std[np.abs(perturb_in_std) < std_limit], bins=25, color='grey')
    print("excluded stds {}".format(np.sum(np.abs(perturb_in_std) >= std_limit)))
    # plt.xlabel("Shifts (std)", size=14)
    # plt.ylabel("# of occurences", size=14)
    plt.gca().annotate("Shifts (std)", xy=(0.5, -0.45), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    plt.gca().annotate("# occurences", xy=(-0.25, 0.0), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline', rotation='vertical')
    plt.legend(fontsize=14, frameon=False)
    plt.ylim(0)
    plt.xlim(-std_limit, std_limit)
    sns.despine()

    fig.subplots_adjust(hspace=0.27)
    fig.subplots_adjust(wspace=0.05)
    # plt.tight_layout()
    # fig.subplots_adjust(left=0.12)
    plt.savefig(fig_path.joinpath('fig_power_analysis.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('fig_power_analysis.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()

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
    plt.close()

    from scipy.stats import pearsonr
    # print(pearsonr(powers, vars))
    # print(pearsonr(powers_ff, vars_ff))
    #
    # print(obs_min, obs_max)
    return powers, vars, powers_ff, vars_ff, perturb_in_std, all_powers, ns


def power_analysis_to_table():
    power_an = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('shifts'), 'rb'))
    local_labs = ['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL', 'UCLA']
    lab_to_num = dict(zip(local_labs, range(len(local_labs))))

    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria)
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

                val = power_an[i, j, 0]
                vals[lab_to_num[lab] * 2] = "$\\infty$" if val > 1000 else np.round(val, 2)
                val = power_an[i, j, 1]
                vals[lab_to_num[lab] * 2 + 1] = "$-\\infty$" if val < -1000 else np.round(val, 2)

            print(formatting_string.format(test_name, reg, *vals))


# qcs = ["pass", "high_noise", "low_yield", "missed_target", "artifacts", "low_trials", "high_lfp", "all"]
# for qc in qcs:
#     print(qc)
#     plot_panel_permutation(qc=qc, n_permut=90000, n_cores=10)
#     plt.title(qc)
#     plt.savefig("temp_{}".format(qc))
#     plt.close()


if __name__ == '__main__':

    # plt.figure(figsize=(16 * 0.75, 9 * 0.75))
    # ax = plt.gca()
    # plt.figure(figsize=(16 * 0.75, 9 * 0.75))
    # ax2 = plt.gca()
    # plot_panel_power_analysis(ax=ax2, ax2=ax)
    # plt.savefig("firing rates plus shifts")
    # plt.show()

    # df = load_dataframe()
    # df_filt = filter_recordings(df, **filtering_criteria)
    # df_filt = df_filt[df_filt['permute_include'] == 1]
    
    # results = pd.DataFrame()
    
    # df_region = df_filt.groupby('region')
    # names = tests.keys()
    # ps = []
    # for test in names:
    #     for i, br in enumerate(BRAIN_REGIONS):
    #         df_br = df_region.get_group(br)
    
    #         vals = df_br.groupby(['subject', 'institute'])[test].mean()
    
    #         labs = vals.index.get_level_values('institute')
    #         subjects = vals.index.get_level_values('subject')
    #         data = vals.values
    #         p = permut_test(data, metric=distribution_dist_approx_max, labels1=np.array(labs), labels2=np.array(subjects), n_permut=10000, n_cores=8)
    #         if p == 0.:
    #             p = 1e-12
    #         results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1],
    #                                                    data={'test': test, 'region': br, 'p_value_permut': p})))
    
    # pickle.dump(results.p_value_permut.values, open(save_data_path(figure='fig_taskmodulation').joinpath('p_values_percent_modulated'), 'wb'))

    plot_main_figure()
    plot_power_analysis()
    # power_analysis_to_table()
    plot_supp_figure()
