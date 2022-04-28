import matplotlib.pyplot as plt
from reproducible_ephys_functions import BRAIN_REGIONS, labs, filter_recordings, save_figure_path, figure_style
from permutation_test import permut_test, permut_dist
from figure5.figure5_load_data import load_dataframe
from figure4.figure4_plot_functions import plot_raster_and_psth
import numpy as np
import pandas as pd
import figrid as fg
import seaborn as sns
from statsmodels.stats.multitest import multipletests

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

def plot_panel_example_neurons(ax1=None, ax2=None, save=True):
    neuron = 614
    pid = 'ce397420-3cd2-4a55-8fd1-5e28321981f4'  # SWC_054
    side = 'right'
    feedback = 'correct'
    align_event = 'stim'

    ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.4, 0.6], fr_bin_size=0.06, align_event=align_event, side=side,
                                   feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 3, 'causal': 1},
                                   zero_line_c=(0, 0.5, 1), labelsize=16, ax=ax1)
    ax[0].set_title(f'Contrast: {side}, {feedback}, Aligned to {align_event}', loc='left')

    if save:
        plt.savefig(fig_path.joinpath(f'figure5_{pid}_neuron{neuron}_align_{align_event}.png'))

    align_event = 'move'
    ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.4, 0.6], fr_bin_size=0.06, align_event=align_event, side=side,
                                   feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 2, 'causal': 1},
                                   zero_line_c='g', labelsize=16, ax=ax2)
    ax[0].set_title(f'Contrast: {side}, {feedback}, Aligned to {align_event}', loc='left')

    if save:
        plt.savefig(fig_path.joinpath(f'figure5_{pid}_neuron{neuron}_align_{align_event}.png'))


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


def plot_panel_permutation(ax=None):

    # Figure 5d permutation tests
    df = load_dataframe()
    df_filt = filter_recordings(df)
    df_filt = df_filt[df_filt['permute_include'] == 1]

    n_permut = 1000
    print("!!!")
    print(n_permut)
    print("!!!")
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

            lab_names, this_n_labs = np.unique(labs, return_counts=True)

            p = permut_test(data, metric=permut_dist, labels1=labs,
                            labels2=subjects, n_permut=n_permut)
            results = results.append(pd.DataFrame(index=[results.shape[0] + 1], data={
                 'test': test, 'region': reg, 'p_value_permut': p}))

    shape = (len(tests.keys()), len(BRAIN_REGIONS))
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


DPI = 400  # if the figure is too big on your screen, lower this number
if __name__ == '__main__':
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
          # 'panel_B': fg.place_axes_on_grid(fig, xspan=[0.6, 1.], yspan=[0.33, 0.63],
          #                                  wspace=0.3),
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
    plot_panel_task_modulated_neurons(specific_tests=['start_to_move'], ax=[ax['panel_C_1'], ax['panel_C_2'], ax['panel_C_3'], ax['panel_C_4'], ax['panel_C_5']], save=False)
    plot_panel_permutation(ax=ax['panel_D'])

    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'd', 'xpos': 0.5, 'ypos': 0.64, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'b', 'xpos': 0, 'ypos': 0.36, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'c', 'xpos': 0, 'ypos': 0.64, 'fontsize': 10, 'weight': 'bold'}]
    fg.add_labels(fig, labels)

    plt.savefig(fig_path.joinpath(f'figure5_combined.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath(f'figure5_combined.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()

# plot_panel_permutation()
