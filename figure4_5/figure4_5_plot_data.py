import matplotlib.pyplot as plt
import numpy as np
import figrid as fg
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs, save_figure_path, figure_style
from figure4_5.figure5_temp_load_data import load_dataframeFig5
from figure4_5.figure4_5_load_data import load_data, load_dataframe
from figure4_5.figure4_plot_functions import plot_raster_and_psth, plot_raster_and_psth_LvsR
import seaborn as sns
import pandas as pd
from statsmodels.stats.multitest import multipletests
import time
from permutation_test import permut_test, distribution_dist_approx


lab_number_map, institution_map, lab_colors = labs()
fig_path = save_figure_path(figure='figure4_5')

# tests = {'trial': 'Trial',
#          'start_to_move': 'Pre move (TW)',
#          'post_stim': 'Post stim',
#          'pre_move': 'Pre move',
#          'pre_move_lr': 'Move LvR',
#          'post_move': 'Post move',
#          'post_reward': 'Post reward',
#          'avg_ff_post_move': 'FanoFactor'}

tests = {'trial': 'Trial post-stim.',
          'start_to_move': 'Reaction period',
          'post_stim': 'Post-stim.',
          'pre_move': 'Pre-move.',
          'pre_move_lr': 'L vs. R pre-move.',
          'post_move': 'Post-move.',
          'post_reward': 'Post-reward',
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
          'panel_A_2': fg.place_axes_on_grid(fig, xspan=[0.08, 0.288], yspan=[0.14, 0.27],
                                             wspace=0.3),
          'panel_B': fg.place_axes_on_grid(fig, xspan=[0.388, 0.631], yspan=[0.045, 0.27],
                                           wspace=0.3),
          'panel_C': fg.place_axes_on_grid(fig, xspan=[0.741, 1.], yspan=[0.045, 0.27],
                                           wspace=0.3),
          'panel_D_1': fg.place_axes_on_grid(fig, xspan=[0.087,  0.277], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_D_2': fg.place_axes_on_grid(fig, xspan=[0.3, 0.49], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_D_3': fg.place_axes_on_grid(fig, xspan=[0.523, 0.713], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_D_4': fg.place_axes_on_grid(fig, xspan=[0.746, .936], yspan=[0.37, 0.58],
                                             wspace=0.3),
          'panel_E_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.66, 0.72],
                                             wspace=0.3),
          'panel_E_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.73, 0.79],
                                             wspace=0.3),
          'panel_E_3': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.8, 0.86],
                                             wspace=0.3),
          'panel_E_4': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.87, 0.93],
                                             wspace=0.3),
          'panel_E_5': fg.place_axes_on_grid(fig, xspan=[0.075, 0.46], yspan=[0.94, 1.],
                                             wspace=0.3),
          'panel_F': fg.place_axes_on_grid(fig, xspan=[0.55, .98], yspan=[0.67, .91],
                                           wspace=0.3)}

    #plot_panel_single_neuron(ax=[ax['panel_A_1'], ax['panel_A_2']], save=False)
    plot_panel_single_neuron_LvsR(ax=[ax['panel_A_1'], ax['panel_A_2']], save=False)
    plot_panel_single_subject(ax=ax['panel_B'], save=False)
    plot_panel_task_modulated_neurons(specific_tests=['pre_move'],
                                      ax=[ax['panel_E_1'], ax['panel_E_2'], ax['panel_E_3'], ax['panel_E_4'], ax['panel_E_5']],
                                      save=False)
    plot_panel_permutation(ax=ax['panel_F'])

    # we have to find out max and min neurons here now, because plots are split
    df = load_dataframe()
    df_filt = filter_recordings(df)
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

    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'b', 'xpos': 0.305, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'c', 'xpos': 0.66, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'd', 'xpos': 0, 'ypos': 0.34, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'e', 'xpos': 0, 'ypos': 0.645, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'f', 'xpos': 0.5, 'ypos': 0.645, 'fontsize': 10, 'weight': 'bold'}]

    fg.add_labels(fig, labels)
    print(f'Saving figures to {fig_path}')
    plt.savefig(fig_path.joinpath('figure4_5_combined.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('figure4_5_combined.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_panel_single_neuron_LvsR(ax=None, save=True):
    # Does not distinguish between contrasts, but distinguishes by side
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

    ax = plot_raster_and_psth_LvsR(pid, neuron, align_event=align_event, feedback=feedback,
                              labelsize=16, ax=ax, **params) #fr_bin_size=0.06, zero_line_c='g',
    ax[0].set_title('Example LP neuron', loc='left')

    if save:
        plt.savefig(fig_path.joinpath(f'figure4_5_{pid}_neuron{neuron}_align_{align_event}.png'))

    #Need to put legend for colorbar/side


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
    side = 'right' #'left' #'all'
    feedback = 'correct' #'all'

    ax = plot_raster_and_psth(pid, neuron, align_event=align_event, side=side, feedback=feedback,
                              labelsize=16, ax=ax, **params) #fr_bin_size=0.06, zero_line_c='g',

    # ax = plot_raster_and_psth(pid, neuron, align_event=align_event, side='left', ax=ax, **params)
    # ax = plot_raster_and_psth(pid, neuron, event_epoch=[-0.2, 0.2], fr_bin_size=0.06, align_event=align_event, side=side,
    #                           feedback=feedback, smoothing='sliding', slide_kwargs_fr={'n_win': 3, 'causal': 1},
    #                           zero_line_c='g', labelsize=16, ax=ax)

    if save:
        plt.savefig(fig_path.joinpath(f'figure4_5_{pid}_neuron{neuron}_align_{align_event}.png'))

    #ax[0].set_title(f'Contrast: {side}, {feedback} choices', loc='left')
    #ax[0].set_title(f'{side} stim., {feedback} choices', loc='left')
    ax[0].set_title('Example LP neuron', loc='left')
    #Need to put legend for colorbar/contrasts


def plot_panel_single_subject(event='move', norm='subtract', smoothing='sliding', ax=None, save=True):
    # Code to plot figure similar to figure 4b
    df = load_dataframe()
    data = load_data(event=event, norm=norm, smoothing=smoothing)

    df_filt = filter_recordings(df)
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
    all_frs_side = all_frs_l #all_frs_r #
    all_frs_side_std = all_frs_l_std #all_frs_r_std #
    
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
    #ax.set_xlim(left=time[0], right=time[-1])
    ax.set_xticks([-0.2, 0, 0.2]) #change this later
    ax.set_xlim(left= -0.2, right= 0.2) #change this later
    #sns.despine(trim=True, ax=ax)

    if save:
        plt.savefig(fig_path.joinpath('figure4_5_example_subject.png'))

    ax.set_title('Example recording in LP', loc='left')


def plot_panel_all_subjects(max_neurons, min_neurons, ax=None, save=True, plotted_regions=BRAIN_REGIONS):
    # Code to plot figure similar to figure 4c
    df = load_dataframe()
    data = load_data(event='move', norm='subtract', smoothing='sliding')

    df_filt = filter_recordings(df)
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
            #Select L vs R side:
            frs_subj = all_frs_l[subj_idx, :]
            #frs_subj = all_frs_r[subj_idx, :]
            if df_subj.iloc[0]['institute'] not in all_present_labs:
                all_present_labs.append(df_subj.iloc[0]['institute'])
            ax[iR].plot(data['time'], np.mean(frs_subj, axis=0), c=lab_colors[df_subj.iloc[0]['institute']],
                        lw=min_lw + ((subj_idx.shape[0] - min_neurons) / (max_neurons - min_neurons)) * max_lw,
                        alpha=0.8)
        ax[iR].set_ylim(bottom=-9, top=13.5)
        ax[iR].set_yticks([-5, 0, 5, 10]) 
        ax[iR].axvline(0, color='k', ls='--')
        ax[iR].spines["right"].set_visible(False)
        ax[iR].spines["top"].set_visible(False)
        #ax[iR].set_xlim(left=data['time'][0], right=data['time'][-1])
        ax[iR].set_xticks([-0.2, 0, 0.2]) #change this later
        ax[iR].set_xlim(left= -0.2, right= 0.2) #change this later
        #sns.despine(trim=True, ax=ax[iR])
        
        if iR >= 1:
            ax[iR].set_yticklabels([])
        else:
            ax[iR].set_ylabel("Baselined firing rate (sp/s)")
            #ax[iR].set_title('Recordings from all labs', loc='left')
            #if len(plotted_regions) != 1:
                #ax[iR].set_ylabel("Baselined firing rate (sp/s)")
                #ax[iR].set_xlabel("Time (s)")
        #ax[iR].set_title(reg)
        if iR == 1 or len(plotted_regions) == 1:
            ax[iR].set_xlabel("Time from movement onset (s)")
            
        if len(plotted_regions) == 1:
            ax[iR].set_title('Recording averages in LP', loc='left')
        else:
            ax[iR].set_title(reg)

        if iR == len(plotted_regions) - 1 and len(plotted_regions) != 1:
            # this is a hack for the legend
            for lab in all_present_labs:
                ax[iR].plot(data['time'], np.zeros_like(data['time']) - 100, c=lab_colors[lab], label=lab)
            leg = ax[iR].legend(frameon=False, bbox_to_anchor=(1, 1), labelcolor='linecolor', handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)

    if save:
        plt.savefig(fig_path.joinpath('figure4_5_all_subjects.png'))


def plot_panel_task_modulated_neurons(specific_tests=None, ax=None, save=True):

    # load dataframe from prev fig. 5 (To be combined with new Fig 4)
    df = load_dataframeFig5()
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
                plt.yticks([0, 1], [0, 1])
                plt.xticks([])
                sns.despine()
                if i == 4:
                    plt.xlabel('Mice')
                elif i==0:
                    plt.title('Proportion of modulated neurons', loc='left')
            else:
                ax[i].bar(np.arange(vals[test].values.shape[0]), vals[test].values, color=colors)
                ax[i].set_ylim(bottom=0, top=1)
                ax[i].set_ylabel(br)
                ax[i].set_yticks([0, 1], [0, 1])
                ax[i].set_xticks([])
                sns.despine()
                if i == 4:
                    ax[i].set_xlabel('Mice')
                elif i==0:
                    ax[i].set_title('Proportion of modulated neurons', loc='left')
        if specific_tests is None:
            plt.suptitle(tests[test], size=22)
        if save:
            plt.savefig(fig_path.joinpath(test))
        


def plot_panel_permutation(ax=None):

    # load dataframe from prev fig. 5 (To be combined with new Fig 4)
    # Prev Figure 5d permutation tests
    df = load_dataframeFig5()
    df_filt = filter_recordings(df, recompute=True)
    df_filt = df_filt[df_filt['permute_include'] == 1]

    df_filt_reg = df_filt.groupby('region')
    results = pd.DataFrame()
    test_names = []
    for test in tests.keys():
        test_names.append(tests[test])
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

            p = permut_test(data, metric=distribution_dist_approx, labels1=labs,
                            labels2=subjects, shuffling='labels1_based_on_2')
            results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1],
                                                      data={'test': test, 'region': reg, 'p_value_permut': p})))

    shape = (len(tests.keys()), len(BRAIN_REGIONS))
    print(results.p_value_permut.values)
    #return

    _, corrected_p_vals, _, _ = multipletests(results.p_value_permut.values, 0.05, method='fdr_bh')
    corrected_p_vals = corrected_p_vals.reshape(shape)
    # corrected_p_vals = results.p_value_permut.values.reshape(shape)

    ax = sns.heatmap(np.log10(corrected_p_vals.T), cmap='RdYlGn', square=True,
                     cbar=True, annot=False, annot_kws={"size": 12}, ax=ax,
                     linewidths=.5, fmt='.2f', vmin=-1.5, vmax=np.log10(1), cbar_kws={"shrink": .7})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.05, 0.1, 0.2, 0.4, 0.8]))
    cbar.set_ticklabels([0.05, 0.1, 0.2, 0.4, 0.8])

    # ax.set(xlabel='', ylabel='', title='Permutation p-values')
    ax.set_yticklabels(BRAIN_REGIONS, va='center', rotation=0)
    ax.set_xticklabels(test_names, rotation=90, ha='right') #rotation=30, ha='right')
    ax.set_title('Task-driven activity: Comparison across labs', loc='left')
    
    return results


# data = data[~np.isnan(data)]
# p = permut_test(data, metric=distribution_dist_approx, labels1=labs,
#                 labels2=subjects, shuffling='labels1_based_on_2', n_permut=10000, plot=True, mark_p=0.004)
# quit()
# results = plot_panel_permutation()
# quit()
if __name__ == '__main__':
    plot_main_figure()
