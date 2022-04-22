import matplotlib.pyplot as plt
from reproducible_ephys_functions import BRAIN_REGIONS, labs, filter_recordings, save_figure_path
from permutation_test import permut_test, permut_dist
from figure5.figure5_load_data import load_dataframe
from figure4.figure4_plot_functions import plot_raster_and_psth
import numpy as np
import pandas as pd

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

def plot_panel_example_neurons():
    neuron = 614
    pid = 'ce397420-3cd2-4a55-8fd1-5e28321981f4'  # SWC_054
    side = 'right'
    feedback = 'correct'
    align_event = 'stim'

    fig, ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.4, 0.6], fr_bin_size=0.06, align_event=align_event, side=side,
                                   feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 2, 'causal': 1},
                                   zero_line_c=(0, 0.5, 1), labelsize=16)
    ax[0].set_title(f'Contrast    {side}, {feedback}, Aligned to {align_event}', loc='left', size=16)

    plt.savefig(fig_path.joinpath(f'figure5_{pid}_neuron{neuron}_align_{align_event}.png'))

    align_event = 'move'
    fig, ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.4, 0.6], fr_bin_size=0.06, align_event=align_event, side=side,
                                   feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 2, 'causal': 1},
                                   zero_line_c='g', labelsize=16)
    ax[0].set_title(f'Contrast    {side}, {feedback}, Aligned to {align_event}', loc='left', size=16)

    plt.savefig(fig_path.joinpath(f'figure5_{pid}_neuron{neuron}_align_{align_event}.png'))


def plot_panel_task_modulated_neurons():

    # load dataframe
    df = load_dataframe()
    df_filt = filter_recordings(df)
    df_filt = df_filt[df_filt['include'] == 1]

    # Group data frame by region
    df_region = df_filt.groupby('region')

    # FIGURE 5c and supplementary figures
    for test in tests.keys():
        for i, br in enumerate(BRAIN_REGIONS):
            plt.subplot(len(BRAIN_REGIONS), 1, i + 1)
            df_br = df_region.get_group(br)

            df_inst = df_br.groupby(['subject', 'institute'], as_index=False)
            vals = df_inst[test].mean().sort_values('institute')
            colors = [lab_colors[col] for col in vals['institute'].values]
            plt.bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
            plt.ylim(bottom=0, top=1)
            plt.ylabel(br)
            plt.xticks([])
            if i == 4:
                plt.xlabel('Mice')
        plt.suptitle(tests[test], size=22)
        plt.savefig(fig_path.joinpath(test))
        plt.close()


def plot_panel_permutation():

    # Figure 5d permutation tests
    df = load_dataframe()
    df_filt = filter_recordings(df)
    df_filt = df_filt[df_filt['permute_include'] == 1]

    n_permut = 10000
    df_filt_reg = df_filt.groupby('region')
    results = pd.DataFrame()
    for test in tests.keys():
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



