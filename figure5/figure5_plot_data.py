import matplotlib.pyplot as plt
from reproducible_ephys_functions import BRAIN_REGIONS, labs, filter_recordings, save_figure_path, figure_style, save_data_path
from permutation_test import permut_test, permut_dist, distribution_dist, distribution_dist_test
from figure5.figure5_load_data import load_dataframe, load_example_neuron
from figure4.figure4_plot_functions import plot_raster_and_psth
import numpy as np
import pandas as pd
import figrid as fg
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import time

_, _, lab_colors = labs()
fig_path = save_figure_path(figure='figure5')

tests = {'trial': 'Trial',
         'start_to_move': 'Pre move (TW)',
         'post_stim': 'Post stim',
         'pre_move': 'Pre move',
         'pre_move_lr': 'Move LvR',
         'post_move': 'Post move',
         'post_reward': 'Post reward',
         'avg_ff_post_move': 'FanoFactor'}


def plot_main_figure():
    DPI = 400  # if the figure is too big on your screen, lower this number
    figure_style()
    fig = plt.figure(figsize=(7, 10.5), dpi=DPI)  # full width figure is 7 inches
    ax = {'panel_A_1_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.45], yspan=[0., 0.15],
                                               wspace=0.3),
          'panel_A_1_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.45], yspan=[0.15, 0.3],
                                               wspace=0.3),
          'panel_A_2_1': fg.place_axes_on_grid(fig, xspan=[0.55, 1.], yspan=[0., 0.15],
                                               wspace=0.3),
          'panel_A_2_2': fg.place_axes_on_grid(fig, xspan=[0.55, 1.], yspan=[0.15, 0.3],
                                               wspace=0.3),
          'panel_B_2': fg.place_axes_on_grid(fig, xspan=[0.6, 1.], yspan=[0.36, 0.63],
                                             wspace=0.3),
          'panel_C_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.66, 0.72],
                                             wspace=0.3),
          'panel_C_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.73, 0.79],
                                             wspace=0.3),
          'panel_C_3': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.8, 0.86],
                                             wspace=0.3),
          'panel_C_4': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.87, 0.93],
                                             wspace=0.3),
          'panel_C_5': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.94, 1],
                                             wspace=0.3),
          'panel_D': fg.place_axes_on_grid(fig, xspan=[0.54, 1.], yspan=[0.66, 1],
                                           wspace=0.3)}

    plot_panel_example_neurons(ax1=[ax['panel_A_1_1'], ax['panel_A_1_2']], ax2=[ax['panel_A_2_1'], ax['panel_A_2_2']], save=False)
    plot_panel_modulation_comparison(ax=ax['panel_B_2'])
    plot_panel_task_modulated_neurons(specific_tests=['start_to_move'],
                                      ax=[ax['panel_C_1'], ax['panel_C_2'], ax['panel_C_3'], ax['panel_C_4'], ax['panel_C_5']],
                                      save=False)
    plot_panel_permutation(ax=ax['panel_D'])

    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'd', 'xpos': 0.5, 'ypos': 0.64, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'b', 'xpos': 0, 'ypos': 0.36, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'c', 'xpos': 0, 'ypos': 0.64, 'fontsize': 10, 'weight': 'bold'}]
    fg.add_labels(fig, labels)

    print(f'Saving figures to {fig_path}')
    plt.savefig(fig_path.joinpath('figure5_combined.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('figure5_combined.pdf'), bbox_inches='tight', pad_inches=0)
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
    fig = plt.figure(figsize=(7, 10.5), dpi=DPI)  # full width figure is 7 inches
    panel_a = task_mod_panel_helper(fig, 'panel_A_', [0.075, 0.45], 0.13, 0.32)
    panel_b = task_mod_panel_helper(fig, 'panel_B_', [0.55, 1.], 0.13, 0.32)
    panel_c = task_mod_panel_helper(fig, 'panel_C_', [0.075, 0.45], 0.46, 0.65)
    panel_d = task_mod_panel_helper(fig, 'panel_D_', [0.55, 1.], 0.46, 0.65)
    panel_e = task_mod_panel_helper(fig, 'panel_E_', [0.075, 0.45], 0.79, 1)
    panel_f = task_mod_panel_helper(fig, 'panel_F_', [0.55, 1.], 0.79, 1)

    plot_panel_task_modulated_neurons(specific_tests=['trial'],
                                      ax=[panel_a['panel_A_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['post_stim'],
                                      ax=[panel_b['panel_B_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['post_move'],
                                      ax=[panel_c['panel_C_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['pre_move'],
                                      ax=[panel_d['panel_D_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['pre_move_lr'],
                                      ax=[panel_e['panel_E_{}'.format(x)] for x in range(1, 6)],
                                      save=False)
    plot_panel_task_modulated_neurons(specific_tests=['post_reward'],
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
    plt.savefig(fig_path.joinpath('figure5_supp.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('figure5_supp.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_panel_example_neurons(ax1=None, ax2=None, save=True):
    neuron = 241 #323 #265 #144 #614
    pid = 'f26a6ab1-7e37-4f8d-bb50-295c056e1062' #'a12c8ae8-d5ad-4d15-b805-436ad23e5ad1' #'36362f75-96d8-4ed4-a728-5e72284d0995'#'31f3e083-a324-4b88-b0a4-7788ec37b191' #'ce397420-3cd2-4a55-8fd1-5e28321981f4'  # SWC_054
    side = 'right'
    feedback = 'correct'
    align_event = 'stim'

    ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.2, 0.2], fr_bin_size=0.06, align_event=align_event, side=side,
                              feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 3, 'causal': 1},
                              zero_line_c=(0, 0.5, 1), labelsize=16, ax=ax1, rxn_time=False) #rxn_time=True) #prev: event_epoch=[-0.4, 0.6]
    ax[0].set_title(f'Contrast: {side}, {feedback}, Aligned to {align_event}', loc='left')

    if save:
        plt.savefig(fig_path.joinpath(f'figure5_{pid}_neuron{neuron}_align_{align_event}.png'))

    align_event = 'move'
    ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.2, 0.2], fr_bin_size=0.06, align_event=align_event, side=side,
                              feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 3, 'causal': 1},
                              zero_line_c='g', labelsize=16, ax=ax2, rxn_time=False) #rxn_time=True)
    ax[0].set_title(f'Contrast: {side}, {feedback}, Aligned to {align_event}', loc='left')

    if save:
        plt.savefig(fig_path.joinpath(f'figure5_{pid}_neuron{neuron}_align_{align_event}.png'))


def plot_panel_modulation_comparison(ax=None, save=True):

    #frs = load_example_neuron()
    #frs = load_example_neuron(id=144, pid = '31f3e083-a324-4b88-b0a4-7788ec37b191')
    frs = load_example_neuron(id=265, pid = '36362f75-96d8-4ed4-a728-5e72284d0995')
    fr_pre_stim = frs[:, 0]
    fr_pre_stim_mean = np.mean(fr_pre_stim)
    fr_pre_stim_std = np.std(fr_pre_stim)
    #fr_pre_stim_sem = fr_pre_stim_std/np.sqrt(np.size(fr_pre_stim))
    x1 = np.ones_like(fr_pre_stim)

    fr_pre_move = frs[:, 1]
    fr_pre_move_mean = np.mean(fr_pre_move)
    fr_pre_move_std = np.std(fr_pre_move)
    #fr_pre_move_sem = fr_pre_move_std/np.sqrt(np.size(fr_pre_move))
    x2 = np.ones_like(fr_pre_move) * 4

    x = np.c_[x1, x2].T
    y = np.c_[fr_pre_stim, fr_pre_move].T

    ax.plot(x, y, 'silver', linestyle="-", linewidth='1', alpha=0.3)
    ax.scatter(x1, fr_pre_stim, facecolors='none', edgecolors=(0, 0.5, 1))
    ax.scatter(x2, fr_pre_move, facecolors='none', edgecolors='g')
    ax.errorbar(x1[0] - 0.15, fr_pre_stim_mean, fr_pre_stim_std, marker="o", markersize=8, c=(0, 0.5, 1), ecolor=(0, 0.5, 1),
                capsize=3)
    ax.errorbar(x2[0] + 0.15, fr_pre_move_mean, fr_pre_move_std, marker="o", markersize=8, c='g', ecolor='g', capsize=3)
    # ax.errorbar(x1[0] - 0.15, fr_pre_stim_mean, fr_pre_stim_sem, marker="o", markersize=8, c=(0, 0.5, 1), ecolor=(0, 0.5, 1),
    #             capsize=3)
    #             capsize=3)
    # ax.errorbar(x2[0] + 0.15, fr_pre_move_mean, fr_pre_move_sem, marker="o", markersize=8, c='g', ecolor='g', capsize=3)
    ax.set_ylabel('Avg. Firing Rate (Sp/s)')
    ax.set_xticks([1, 4])
    ax.set_xticklabels(['Pre-stim', 'Pre-movement'])
    ax.set_title('Example task modulated neuron')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_panel_task_modulated_neurons(specific_tests=None, ax=None, save=True):

    # load dataframe
    df = load_dataframe()
    df_filt = filter_recordings(df)
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
                plt.xticks([])
                if i == 4:
                    plt.xlabel('Mice')
            else:
                ax[i].bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
                ax[i].set_ylim(bottom=0, top=1)
                ax[i].set_ylabel(br)
                ax[i].set_xticks([])
                if i == 4:
                    ax[i].set_xlabel('Mice')
        if specific_tests is None:
            plt.suptitle(tests[test], size=22)
        if save:
            plt.savefig(fig_path.joinpath(test))


def plot_lab_means(data, labs, subjects):
    # takes input like the permutation test and plots it
    counter = 0
    for l in np.unique(labs):
        points = data[labs == l]
        plt.plot(np.arange(counter, counter + len(points)), points, 'o', color=lab_colors[l])
        plt.plot([counter, counter + len(points) - 1], [np.mean(points), np.mean(points)])
        counter += len(points)
    plt.plot([0, counter], [np.mean(data), np.mean(data)])


def plot_panel_permutation(ax=None):

    # Figure 5d permutation tests
    df = load_dataframe()
    df_filt = filter_recordings(df)
    df_filt = df_filt[df_filt['permute_include'] == 1]

    df_filt_reg = df_filt.groupby('region')
    results = pd.DataFrame()
    test_names = []
    for test in tests.keys():
        test_names.append(tests[test])
        for reg in BRAIN_REGIONS:
            df_reg = df_filt_reg.get_group(reg)
            vals = df_reg.groupby(['institute', 'subject'])[test].mean()
            labs = vals.index.get_level_values('institute')
            subjects = vals.index.get_level_values('subject')
            data = vals.values

            # lab_names, this_n_labs = np.unique(labs, return_counts=True)  # what is this for?

            a = time.time()
            p = permut_test(data, metric=distribution_dist_test, labels1=labs,
                            labels2=subjects)
            print(time.time() - a)
            results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1],
                                                      data={'test': test, 'region': reg, 'p_value_permut': p})))

    shape = (len(tests.keys()), len(BRAIN_REGIONS))
    print(results.p_value_permut.values)
    return
    # return results
    _, corrected_p_vals, _, _ = multipletests(results.p_value_permut.values, 0.05, method='fdr_bh')
    corrected_p_vals = corrected_p_vals.reshape(shape)
    # corrected_p_vals = results.p_value_permut.values.reshape(shape)

    sns.heatmap(np.log10(corrected_p_vals.T), cmap='RdYlGn', square=True,
                cbar=True, annot=False, annot_kws={"size": 12}, ax=ax,
                linewidths=.5, fmt='.2f', vmin=-1.5, vmax=np.log10(1), cbar_kws={"shrink": .7})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.05, 0.1, 0.2, 0.4, 0.8]))
    cbar.set_ticklabels([0.05, 0.1, 0.2, 0.4, 0.8])

    ax.set(xlabel='', ylabel='', title='Permutation p-values')
    ax.set_yticklabels(BRAIN_REGIONS, va='center', rotation=0)
    ax.set_xticklabels(test_names, rotation=30, ha='right')

    return results


if __name__ == '__main__':
    plot_main_figure()
    plot_supp_figure()
