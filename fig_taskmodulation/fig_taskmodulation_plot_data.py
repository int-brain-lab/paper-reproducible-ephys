import figrid as fg
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
import numpy as np
import pickle
from scipy.stats import sem
import seaborn as sns

from fig_taskmodulation.fig_taskmodulation_load_data import load_data, load_dataframe, tests, filtering_criteria
from fig_taskmodulation.fig_taskmodulation_plot_functions import plot_raster_and_psth, plot_raster_and_psth_LvsR
from reproducible_ephys_functions import (filter_recordings, BRAIN_REGIONS, REGION_RENAME, LAB_MAP, save_figure_path,
                                          figure_style, save_data_path, get_label_pos, get_row_coord,
                                          plot_vertical_institute_legend, plot_horizontal_institute_legend)

np.set_printoptions(suppress=True)
lab_number_map, institution_map, lab_colors = LAB_MAP()
fig_path = save_figure_path(figure='fig_taskmodulation')

PRINT_PIDS = False
PRINT_INFO = False

shortened_tests = {'post_stim': 'Stimulus',
                   'post_move': 'Movement',
                   'start_to_move': 'Reaction period',
                   'pre_move': 'Move initiation',
                   'post_reward': 'Reward',
                   'pre_move_lr': 'L vs R move',
                   'avg_ff_post_move': 'Fano Factor'}


def plot_main_figure():

    height = 9
    width = 7

    yspans = get_row_coord(height, [1, 1, 4], hspace=0.8, pad=0.3)
    xspan_row1 = get_row_coord(width, [1, 1, 1], pad=0.5)
    xspan_row2 = get_row_coord(width, [3, 3, 3, 3, 1], hspace=0.1, pad=0.5)
    xspan_row3 = get_row_coord(width, [1, 1], pad=0.5)

    xspan_inset = get_row_coord(width, [1, 3], hspace=0.2, pad=0, span=xspan_row3[1])
    yspan_inset = get_row_coord(height, [1, 3], pad=0, span=yspans[2])

    figure_style()
    fig = plt.figure(figsize=(width, height), dpi=150)
    hxcoords = [xspan_row3[1][0] + 0.05, xspan_row3[1][-1] - 0.05]
    hycoords = [yspan_inset[1][0], yspan_inset[1][-1] - 0.05]
    axes = {'A': fg.place_axes_on_grid(fig, xspan=xspan_row1[0], yspan=yspans[0], dim=[2, 1], hspace=0.1),
            'B': fg.place_axes_on_grid(fig, xspan=xspan_row1[1], yspan=yspans[0]),
            'C': fg.place_axes_on_grid(fig, xspan=xspan_row1[2], yspan=yspans[0]),
            'D_VISa': fg.place_axes_on_grid(fig, xspan=xspan_row2[0], yspan=yspans[1]),
            'D_CA1': fg.place_axes_on_grid(fig, xspan=xspan_row2[1], yspan=yspans[1]),
            'D_DG': fg.place_axes_on_grid(fig, xspan=xspan_row2[2], yspan=yspans[1]),
            'D_PO': fg.place_axes_on_grid(fig, xspan=xspan_row2[3], yspan=yspans[1]),
            'D_labs': fg.place_axes_on_grid(fig, xspan=xspan_row2[4], yspan=yspans[1]),
            'E': fg.place_axes_on_grid(fig, xspan=xspan_row3[0], yspan=yspans[2], dim=[7, 1], hspace=0.1),
            'F': fg.place_axes_on_grid(fig, xspan=xspan_inset[0], yspan=yspan_inset[0]),
            'G': fg.place_axes_on_grid(fig, xspan=xspan_inset[1], yspan=yspan_inset[0]),
            'H': fg.place_axes_on_grid(fig, xspan=hxcoords, yspan=hycoords)
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width,xspan_row1[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspan_row1[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspan_row1[2][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width, xspan_row2[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width, xspan_row3[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'f', 'xpos': get_label_pos(width, xspan_inset[0][0], pad=0.5),
               'ypos': get_label_pos(height,yspan_inset[0][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'g', 'xpos': get_label_pos(width, xspan_inset[1][0]),
               'ypos': get_label_pos(height, yspan_inset[0][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'h', 'xpos': get_label_pos(width, hxcoords[0], pad=0.5),
               'ypos': get_label_pos(height, hycoords[0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'}]

    fg.add_labels(fig, labels)

    # Find the maximum and minimum number of neurons in any region in any institute needed for panels C and D
    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria)
    df_filt = df_filt[df_filt['permute_include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby(['region', 'institute'])
    max_neurons = df_filt_reg['cluster_ids'].count().max()
    min_neurons = df_filt_reg['cluster_ids'].count().min()

    # Panel A
    plot_panel_single_neuron_LvsR(ax=axes['A'], save=False)
    # Panel B
    plot_panel_single_subject(ax=axes['B'], save=False)
    # Panel C
    plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons, ax=[axes['C']],
                            save=False, plotted_regions=['LP'])
    # Panel D
    all_labs = plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons,
                                       ax=[axes['D_VISa'], axes['D_CA1'], axes['D_DG'], axes['D_PO']], save=False,
                                       plotted_regions=[r for r in BRAIN_REGIONS if r != 'LP'])
    # Adjust the xaxis label so they are shared by all plots in D
    axes['D_CA1'].xaxis.set_label_coords(1.15, -0.2)

    # Remove the axis for location that shows lab names
    axes['D_labs'].set_axis_off()
    plot_vertical_institute_legend(all_labs, axes['D_labs'])

    # Panel E
    axes['E'][0].set_axis_off()
    axes['E'][1].set_axis_off()
    plot_panel_task_modulated_neurons(specific_tests=['pre_move'], ax=axes['E'][2:], save=False)

    # Panels G and F
    plot_panel_power_analysis(ax=axes['G'], ax2=axes['F'])

    # Panel H
    plot_panel_permutation(ax=axes['H'])

    print(f'Saving figures to {fig_path}')
    adjust = 0.3
    fig.subplots_adjust(top=1 - adjust / height, bottom=(adjust + 0.2) / height, left=(adjust)/ width,
                        right=1 - adjust / width)
    plt.savefig(fig_path.joinpath('fig_taskmodulation.svg'))
    plt.savefig(fig_path.joinpath('fig_taskmodulation.pdf'))
    plt.close()

def plot_supp_figure():
    figure_style()
    width = 7
    height = 9
    fig = plt.figure(figsize=(width, height))

    padx = 0.4

    xspans = get_row_coord(width, [1, 1], pad=padx)
    yspans = get_row_coord(height, [7, 7, 7, 1], hspace=[0.5, 0.5, 0.4], pad=0.3)
    xspan_all = get_row_coord(width, [1])
    ax = {'A': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=[7, 1], hspace=0.8),
          'B': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0], dim=[7, 1], hspace=0.8),
          'C': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1], dim=[7, 1], hspace=0.8),
          'D': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1], dim=[7, 1], hspace=0.8),
          'E': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[2], dim=[7, 1], hspace=0.8),
          'F': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[2], dim=[7, 1], hspace=0.8),
          'G': fg.place_axes_on_grid(fig, xspan=xspan_all[0], yspan=yspans[3])
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width,xspans[0][0], pad=padx),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspans[1][0], pad=padx),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspans[0][0], pad=padx),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width, xspans[1][0], pad=padx),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width, xspans[0][0], pad=padx),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'f', 'xpos': get_label_pos(width, xspans[1][0], pad=padx),
               'ypos': get_label_pos(height,yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'}]


    fg.add_labels(fig, labels)

    plot_panel_task_modulated_neurons(specific_tests=['post_stim'], ax=ax['A'][2:], save=False)
    ax['A'][0].set_title('Stimulus vs pre stimulus test')

    plot_panel_task_modulated_neurons(specific_tests=['post_move'], ax=ax['B'][2:], save=False)
    ax['B'][0].set_title('Movement vs pre stimulus test')

    plot_panel_task_modulated_neurons(specific_tests=['start_to_move'], ax=ax['C'][2:], save=False)
    ax['C'][0].set_title('Reaction period vs pre stimulus test')

    plot_panel_task_modulated_neurons(specific_tests=['pre_move'], ax=ax['D'][2:], save=False)
    ax['D'][0].set_title('Movement initiation vs pre stimulus test')

    plot_panel_task_modulated_neurons(specific_tests=['post_reward'], ax=ax['E'][2:], save=False)
    ax['E'][0].set_title('Reward vs pre stimulus test')

    inst = plot_panel_task_modulated_neurons(specific_tests=['pre_move_lr'], ax=ax['F'][2:],save=False)
    ax['F'][0].set_title('Left vs right movement test')

    ax['G'].set_axis_off()
    plot_horizontal_institute_legend(inst, ax['G'])

    for a, v in ax.items():
        if a == 'G':
            continue
        if a not in ['E', 'F']:
            v[-1].set_xlabel('')
        else:
            v[-1].set_xlabel('Percentage modulated neurons')
        v[0].set_axis_off()
        v[1].set_axis_off()
        # Shift the VIs/am label
        v[2].yaxis.set_label_coords(-0.02, 1.02)

    print(f'Saving figures to {fig_path}')
    adjust = 0.3
    fig.subplots_adjust(top=1 - adjust / height, bottom=0.2 / height, left=adjust/ width,
                        right=1 - adjust / width)
    plt.savefig(fig_path.joinpath('figure_taskmodulation_supp.svg'))
    plt.savefig(fig_path.joinpath('figure_taskmodulation_supp.pdf'))
    plt.close()


def plot_panel_single_neuron(ax=None, save=False, pid='f26a6ab1-7e37-4f8d-bb50-295c056e1062', neuron=241):

    # Code to plot figure similar to figure 4a; plots separately for each contrast
    params = {'smoothing': 'sliding',
              'fr_bin_size': 0.06,
              'event_epoch': [-0.2, 0.2],
              'slide_kwargs_fr': {'n_win': 3, 'causal': 1}}
    align_event = 'move'
    feedback = 'correct'  #'all'
    contrasts = (1, 0.25, 0.125, 0.0625) # excluding 0 contrasts

    ax = plot_raster_and_psth(pid, neuron, align_event=align_event, feedback=feedback,
                              labelsize=16, ax=ax, contrasts=contrasts, **params)
    ax[0].set_title('Example LP neuron')

    if save:
        plt.savefig(fig_path.joinpath(f'fig_taskmodulation_{pid}_neuron{neuron}_align_{align_event}.png'))


def plot_panel_single_neuron_LvsR(ax=None, save=False, pid='f26a6ab1-7e37-4f8d-bb50-295c056e1062', neuron=650):
    # Does not distinguish between contrasts, but distinguishes by side

    params = {'smoothing': 'sliding',
              'fr_bin_size': 0.06,
              'event_epoch': [-0.2, 0.2],  # [-0.3, 0.22],
              'slide_kwargs_fr': {'n_win': 3, 'causal': 1}}

    align_event = 'move'
    side = 'right'
    feedback = 'correct'

    ax = plot_raster_and_psth_LvsR(pid, neuron, align_event=align_event, side=side, feedback=feedback,
                              labelsize=16, ax=ax, **params)
    if PRINT_INFO:
        print('Figure 4a:')
        print(f'N_inst: {1}, N_sess: {1}, '
              f'N_mice: {1}, N_cells: {1}')

    if save:
        plt.savefig(fig_path.joinpath(f'fig_taskmodulation_{pid}_neuron{neuron}_align_{align_event}.png'))

    ax[0].set_title('Example LP neuron')


def plot_panel_single_subject(event='move', norm='subtract', smoothing='sliding', ax=None, save=False):
    # Code to plot figure similar to figure 4b
    df = load_dataframe()
    data = load_data(event=event, norm=norm, smoothing=smoothing)
    df_filt = filter_recordings(df, **filtering_criteria)

    all_frs_l = data['all_frs_l'][df_filt['include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['include'] == 1]
    all_frs_l_std = data['all_frs_l_std'][df_filt['include'] == 1]
    all_frs_r_std = data['all_frs_r_std'][df_filt['include'] == 1]
    df_filt = df_filt[df_filt['include'] == 1].reset_index()

    if ax is None:
        fig, ax = plt.subplots()
    subject = 'DY_018'
    region = 'LP'
    idx = df_filt.loc[(df_filt['region'] == region) & (df_filt['subject'] == subject)].index
    lab = df_filt.loc[idx[0]].institute
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
    ax.plot(time, fr_mean, c=lab_colors[lab], lw=1.5)
    propagated_error = np.sqrt(propagated_error) / idx.shape[0]
    ax.fill_between(time, fr_mean - propagated_error, fr_mean + propagated_error, color=lab_colors[lab], alpha=0.25)
    ax.axvline(0, color='k', ls='--')

    ax.set_xlabel("Time from movement onset (s)")
    ax.set_ylabel("Baselined firing rate (spikes/s)")
    ax.set_xlim(left=time[0], right=time[-1])
    ax.set_xticks([-0.15, 0, 0.15], [-0.15, 0, 0.15]) #change this later
    ax.set_title('Example recording in LP')

    if PRINT_INFO:
        print('Figure 4b:')
        print(f'N_inst: {1}, N_sess: {1}, '
              f'N_mice: {1}, N_cells: {len(idx)}')

    if save:
        plt.savefig(fig_path.joinpath('fig_taskmodulation_example_subject.png'))


def plot_panel_all_subjects(max_neurons, min_neurons, ax=None, save=True, plotted_regions=BRAIN_REGIONS):

    # Code to plot figure similar to figure 4c and 4d
    df = load_dataframe()
    data = load_data(event='move', norm='subtract', smoothing='sliding')

    df_filt = filter_recordings(df, **filtering_criteria)
    all_frs_l = data['all_frs_l'][df_filt['permute_include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['permute_include'] == 1]
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
            fr = np.mean(frs_inst, axis=0)
            err = sem(frs_inst, axis=0) / 2
            # ax[iR].errorbar(data['time'], np.mean(frs_inst, axis=0), yerr=sem(frs_inst, axis=0) / 2, c=lab_colors[inst],
            #                 #lw=min_lw + ((inst_idx.shape[0] - min_neurons) / (max_neurons - min_neurons)) * max_lw,
            #                 lw=1,
            #                 alpha=0.8)
            ax[iR].plot(data['time'], fr, c=lab_colors[inst])
            ax[iR].fill_between(data['time'], fr - err, fr + err, color=lab_colors[inst], alpha=0.25)
        ax[iR].set_ylim(bottom=-4, top=8)
        ax[iR].set_yticks([-4, 0, 4, 8])
        ax[iR].axvline(0, color='k', ls='--')
        ax[iR].set_xlim(left=data['time'][0], right=data['time'][-1])
        ax[iR].set_xticks([-0.15, 0, 0.15], [-0.15, 0, 0.15])  # change this later
        if iR >= 1:
            ax[iR].set_yticklabels([])
        else:
            ax[iR].set_ylabel("Baselined firing rate (spikes/s)")

        if len(plotted_regions) == 1:
            ax[iR].set_title('Recording averages in LP')
        else:
            ax[iR].set_title(REGION_RENAME[reg])

        if iR == 1 or len(plotted_regions) == 1:
            ax[iR].set_xlabel("Time from movement onset (s)")

        if PRINT_INFO:
            print(f'Figure 4c/d: {reg}')
            print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
                  f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')

    if save:
        plt.savefig(fig_path.joinpath('fig_taskmodulation_all_subjects.png'))

    return all_present_labs


def plot_panel_task_modulated_neurons(specific_tests=None, ax=None, save=True):

    # load dataframe from prev fig. 5 (To be combined with new Fig 4)
    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria)
    df_filt = df_filt[df_filt['permute_include'] == 1]
    inst = df_filt.institute.unique()
    inst.sort()

    if PRINT_PIDS:
        json.dump(list(np.unique(df_filt['pid'])), open("panel e", 'w'))
        return

    # Group data frame by region
    df_region = df_filt.groupby('region')
    lab_to_num = dict(zip(inst, list(range(len(inst)))))

    names = tests.keys() if specific_tests is None else specific_tests
    # FIGURE 5c and supplementary figures
    for test in names:
        for i, br in enumerate(BRAIN_REGIONS):

            df_br = df_region.get_group(br)
            df_inst = df_br.groupby(['subject', 'eid', 'institute'], as_index=False)
            if PRINT_INFO:
                print(f'Figure 4e/ Figure 4 supp 1: {br}: {test}')
                info = df_inst[test].count().sort_values('institute')
                print(f'N_inst: {info.institute.nunique()}, N_sess: {info.eid.nunique()}, '
                      f'N_mice: {info.subject.nunique()}, N_cells: {info[test].sum()}')

            vals = df_inst[test].mean().sort_values('institute')
            colors = [lab_colors[col] for col in vals['institute'].values]
            positions = [lab_to_num[lab] + np.random.rand() * 0.1 - 0.05 for lab in vals['institute'].values]

            if ax is None:
                plt.subplot(len(BRAIN_REGIONS), 1, i + 1)
                plt.bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
                plt.ylim(bottom=0, top=1)
                plt.ylabel(REGION_RENAME[br])
                plt.yticks([0, 1], [0, 1])
                plt.xticks([])
                sns.despine()
                if i == 4:
                    plt.xlabel('Mice')
                elif i == 0:
                    plt.title('Percentage modulated neurons', loc='left')
            else:
                for lab in vals['institute'].values:
                    mean = vals[test].values[vals['institute'].values == lab].mean()
                    stan_err = sem(vals[test].values[vals['institute'].values == lab])
                    ax[i].plot([mean, mean], [lab_to_num[lab]-0.2, lab_to_num[lab]+0.2], color=lab_colors[lab], lw=1)
                    ax[i].plot([mean-stan_err / 2, mean+stan_err / 2], [lab_to_num[lab], lab_to_num[lab]], lw=0.5, color=lab_colors[lab])
                ax[i].scatter(vals[test].values, positions, color=colors, s=1)
                ax[i].set_xlim(-0.05, 1.05)
                ax[i].set_ylabel(REGION_RENAME[br])
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                sns.despine()

                if i == 4:
                    ax[i].set_xticks([0, 1], [0, 100])  # we are plotting percentages
                    ax[i].set_xlabel('Percentage modulated neurons ({} test)'.format(shortened_tests[test]))
        if specific_tests is None:
            plt.suptitle(tests[test], size=22)
        if save:
            plt.savefig(fig_path.joinpath(test))

    return inst


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
    cbar.set_label('p-value', rotation=270, labelpad=8)

    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    ax.set_yticklabels([REGION_RENAME[br] for br in BRAIN_REGIONS] + [REGION_RENAME[br] for br in BRAIN_REGIONS], va='center', rotation=0)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    ax.set_xticklabels(test_names, ha='right', rotation=45)

    ax.set_title("Percentage modulated neurons", size=7)
    ax.annotate("Distribution of firing rate modulations", (0.5, 0.485), horizontalalignment='center', xycoords='axes fraction', size=7)

    return p_vals


def plot_panel_power_analysis(ax, ax2):

    significant_disturbances = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('shifts'), 'rb'))
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

    def configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--'):
        parts['bodies'][0].set_zorder(zo)
        parts['bodies'][0].set_facecolor(fc)
        parts['bodies'][0].set_edgecolor(ec)
        parts['bodies'][0].set_linestyles(ls)


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

            df_reg = df_reg[~np.isnan(data)]
            data = data[~np.isnan(data)]
            labs = df_reg['institute'].values

            if PRINT_INFO:
                print('Figure 4f/g:')
                print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
                      f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')


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
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] - 1, positions=[0], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] + 2, positions=[0], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='red', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] - 2, positions=[0], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='red', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 1, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 1, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 2, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 2, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 3, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 3, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='grey', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 + 4, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='red', zo=-1, ls='--')

                    parts = ax2.violinplot(data[labs == lab] * 1.4 - 4, positions=[dist_between_violins], showmeans=False, showextrema=False)
                    configure_parts(parts, fc='white', ec='red', zo=-1, ls='--')

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

            ax2.set_ylabel('Firing rate modulation (spikes/s)')
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

    figure_style()
    width = 7
    height = 8
    fig = plt.figure(figsize=(width, height))

    padx = 0.5
    xspans = get_row_coord(width, [1], pad=padx)
    xspans_row2 = get_row_coord(width, [1, 1], hspace=1.4, span=[0.1, 0.9], pad=padx)
    yspans = get_row_coord(height, [21, 1, 6], hspace=[0.3, 0.5], pad=0.3)

    axs = {'A': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=[7, 5], wspace=0.1, hspace=0.2),
          'A_1': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
          'B': fg.place_axes_on_grid(fig, xspan=xspans_row2[0], yspan=yspans[2]),
          'C': fg.place_axes_on_grid(fig, xspan=xspans_row2[1], yspan=yspans[2]),
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans[0][0], pad=padx),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspans_row2[0][0], pad=padx),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspans_row2[1][0], pad=padx),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},]

    fg.add_labels(fig, labels)

    for jj, test in enumerate(tests.keys()):
        for ii, reg in enumerate(BRAIN_REGIONS):

            ax = axs['A'][jj][ii]

            # plt.subplot(8, 5, ii + jj * 5 + 1)
            df_reg = df_filt_reg.get_group(reg)
            i += 1

            if test == 'avg_ff_post_move':
                data = df_reg[test].values
            else:
                data = df_reg['mean_fr_diff_{}'.format(test)].values

            df_reg = df_reg[~np.isnan(data)]
            data = data[~np.isnan(data)]
            labs = df_reg['institute'].values

            if PRINT_INFO:
                print(f'Figure 4 supp2 a: {reg} : {test}')
                print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
                      f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')

            ax.annotate("p={:.3f}".format(p_values[i]), xy=(0.5, 0.95), xycoords='axes fraction',
                        size='6', ha='center', va='bottom')
            if significant_disturbances[i, 0, 0] == 0 and significant_disturbances[i, 0, 1] == 0:
                if jj == 6:
                    ax.plot([-0.3, 6.3], [ff_min_y, ff_max_y], 'k')
                    ax.plot([-0.3, 6.3], [ff_max_y, ff_min_y], 'k')
                else:
                    ax.plot([-0.3, 6.3], [min_y, max_y], 'k')
                    ax.plot([-0.3, 6.3], [max_y, min_y], 'k')

            for j, lab in enumerate(np.unique(labs)):
                if np.sum(labs == lab) == 0:
                    continue

                lab_mean = data[labs == lab].mean()
                ax.plot([lab_to_num[lab] - 0.3, lab_to_num[lab] + 0.3], [lab_mean, lab_mean], color=lab_colors[lab])

                parts = ax.violinplot(data[labs == lab], positions=[lab_to_num[lab]], showextrema=False)  # , showmeans=True)
                # print("{}, {}, {}".format(lab, lab_to_num[lab], np.min(data[labs == lab])))
                parts['bodies'][0].set_facecolor(lab_colors[lab])
                parts['bodies'][0].set_edgecolor(lab_colors[lab])

                # parts['cmeans'].set_color('k') # this can be used to check whether the means align -> whether the datasets are assigned correctly

                val = significant_disturbances[i, j, 0]

                if not (significant_disturbances[i, 0, 0] == 0 and significant_disturbances[i, 0, 1] == 0):
                    perturb_in_std.append(val / np.std(data[labs == lab]))
                    if test == 'avg_ff_post_move':
                        powers_ff.append(val)
                        vars_ff.append(np.std(data[labs == lab]) / np.sqrt(np.sum(labs == lab)))
                    else:
                        powers.append(val)
                        vars.append(np.std(data[labs == lab]) / np.sqrt(np.sum(labs == lab)))

                if test == 'avg_ff_post_move':
                    ax.axhline(1, color='grey', alpha=1/3, zorder=0)
                else:
                    ax.axhline(0, color='grey', alpha=1/3, zorder=0)

                temp_color = lab_colors[lab] if val < 1000 else 'red'
                if isinstance(temp_color, str):
                    val = max_y - lab_mean
                    # print(ii + jj * 8 + 1)
                ax.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                obs_max = max(obs_max, lab_mean + val)
                val = significant_disturbances[i, j, 1]

                if not (significant_disturbances[i, 0, 0] == 0 and significant_disturbances[i, 0, 1] == 0):
                    perturb_in_std.append(val / np.std(data[labs == lab]))
                    if test == 'avg_ff_post_move':
                        powers_ff[-1] -= val
                    else:
                        powers[-1] -= val

                temp_color = lab_colors[lab] if val > -1000 else 'red'
                if isinstance(temp_color, str):
                    val = min_y - lab_mean
                    # print(ii + jj * 8 + 1)
                ax.plot([lab_to_num[lab] + perturbation_shift, lab_to_num[lab] + perturbation_shift], [lab_mean, lab_mean + val], color=temp_color)
                obs_min = min(obs_min, lab_mean + val)
            ax.set_xlim(-0.3, (len(lab_to_num) - 1) * dist_between_violins + .36)
            sns.despine()
            if jj == 6:
                if ii == 0:
                    ax.set_ylabel('Fano factor')
                    ax.yaxis.set_label_coords(-0.25, 0.5)

                ax.set_ylim(ff_min_y, ff_max_y)
                ax.set_xticks([])
                ax.set_xlabel('Labs')
            else:
                ax.set_ylim(min_y, max_y)
                ax.set_xticks([])

            if ii != 0 and ii != 6:
                ax.set_yticks([])

            if ii == 4:
                test_name = shortened_tests[test]
                if test_name == 'Reaction period':
                    test_name = 'Reaction \n period'

                ax.annotate(test_name, xy=(1.1, 0.5), xycoords='axes fraction', size='8', ha='left', va='center', rotation='vertical')

            if jj == 0:
                ax.set_title(REGION_RENAME[reg], pad=20)

            if jj == 3 and ii == 0:
                ax.set_ylabel('Firing rate modulation (spikes/s)')
                ax.yaxis.set_label_coords(-0.25, 1.1)


    axs['A_1'].set_axis_off()
    inst = df_filt.institute.unique()
    plot_horizontal_institute_legend(inst, axs['A_1'], offset=0.15)

    if PRINT_INFO:
        print(f'Figure 4 supp2 b/c')
        print(f'N_inst: {df_filt.institute.nunique()}, N_sess: {df_filt.eid.nunique()}, '
              f'N_mice: {df_filt.subject.nunique()}, N_cells: {len(df_filt)}')

    axs['B'].scatter(powers, vars, color='blue', label="Firing modulation", s=0.15)
    axs['B'].scatter(powers_ff, vars_ff, color='blue', label="Fano factor", s=0.15)
    axs['B'].set_xlabel("Minimal shift magnitude (sp/s)")
    axs['B'].set_ylabel(r"Std / $\sqrt{N}$")

    std_limit = 3
    perturb_in_std = np.array(perturb_in_std)
    assert np.sum(perturb_in_std == 0) == 0, "Perturbations of significant tests in histogram"
    axs['C'].hist(perturb_in_std[np.abs(perturb_in_std) < std_limit], bins=25, color='grey')
    print("excluded stds {}".format(np.sum(np.abs(perturb_in_std) >= std_limit)))
    axs['C'].set_xlabel("Shifts (std)")
    axs['C'].set_ylabel("Number of occurences")
    axs['C'].set_xlim(-std_limit, std_limit)

    print(f'Saving figures to {fig_path}')
    adjust = 0.3
    fig.subplots_adjust(top=1 - adjust / height, bottom=(adjust + 0.2) / height, left=(adjust) / width,
                        right=1 - (adjust + 0.2) / width)
    plt.savefig(fig_path.joinpath('figure_power_analysis.svg'))
    plt.savefig(fig_path.joinpath('figure_power_analysis.pdf'))
    plt.close()


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


if __name__ == '__main__':


    plot_main_figure()
    plot_power_analysis()
    # power_analysis_to_table()
    plot_supp_figure()
