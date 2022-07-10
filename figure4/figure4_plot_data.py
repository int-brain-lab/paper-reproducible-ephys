import matplotlib.pyplot as plt
import numpy as np
import figrid as fg
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs, save_figure_path, figure_style
from figure4.figure4_load_data import load_data, load_dataframe
from figure4.figure4_plot_functions import plot_raster_and_psth
import seaborn as sns

lab_number_map, institution_map, lab_colors = labs()
fig_path = save_figure_path(figure='figure4')


def plot_main_figure():
    DPI = 400  # if the figure is too big on your screen, lower this number
    figure_style()
    fig = plt.figure(figsize=(7, 10.5), dpi=DPI)  # full width figure is 7 inches
    ax = {'panel_A_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.323], yspan=[0., 0.15],
                                             wspace=0.3),
          'panel_A_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.323], yspan=[0.15, 0.3],
                                             wspace=0.3),
          'panel_B': fg.place_axes_on_grid(fig, xspan=[0.423, 0.671], yspan=[0., 0.3],
                                           wspace=0.3),
          'panel_C': fg.place_axes_on_grid(fig, xspan=[0.751, 1.], yspan=[0., 0.3],
                                           wspace=0.3),
          'panel_D_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.27375], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_D_2': fg.place_axes_on_grid(fig, xspan=[0.28375, 0.4825], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_D_3': fg.place_axes_on_grid(fig, xspan=[0.4925, 0.69125], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_D_4': fg.place_axes_on_grid(fig, xspan=[0.70125, 0.9], yspan=[0.38, 0.65],
                                             wspace=0.3)}
          # 'panel_D': fg.place_axes_on_grid(fig, xspan=[0.075, 1.], yspan=[0.7, 1],
          #                                  wspace=0.3)}
    plot_panel_single_neuron(ax=[ax['panel_A_1'], ax['panel_A_2']], save=False)
    plot_panel_single_subject(ax=ax['panel_B'], save=False)

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
    D_regions = [reg for reg in BRAIN_REGIONS if reg != 'CA1']
    plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons, ax=[ax['panel_C']], save=False, plotted_regions=['CA1'])
    plot_panel_all_subjects(max_neurons=max_neurons, min_neurons=min_neurons, ax=[ax['panel_D_1'], ax['panel_D_2'], ax['panel_D_3'], ax['panel_D_4']], save=False, plotted_regions=D_regions)

    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'b', 'xpos': 0.35, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'c', 'xpos': 0.671, 'ypos': 0, 'fontsize': 10, 'weight': 'bold'},
              {'label_text': 'd', 'xpos': 0, 'ypos': 0.36, 'fontsize': 10, 'weight': 'bold'}]

    fg.add_labels(fig, labels)
    print(f'Saving figures to {fig_path}')
    plt.savefig(fig_path.joinpath('figure4_combined.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_path.joinpath('figure4_combined.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_panel_single_neuron(ax=None, save=True):
    # Code to plot figure similar to figure 4a
    pid = 'f26a6ab1-7e37-4f8d-bb50-295c056e1062'
    neuron = 386
    align_event = 'move'
    params = {'smoothing': 'sliding',
              'fr_bin_size': 0.04,
              'event_epoch': [-0.3, 0.22],
              'slide_kwargs_fr': {'n_win': 3, 'causal': 1}}
    ax = plot_raster_and_psth(pid, neuron, align_event=align_event, side='left', ax=ax, **params)

    if save:
        plt.savefig(fig_path.joinpath(f'figure4_{pid}_neuron{neuron}_align_{align_event}.png'))


def plot_panel_single_subject(event='move', norm='subtract', smoothing='kernel', ax=None, save=True):
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
    subject = 'SWC_054'
    region = 'CA1'
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
    propagated_error = np.zeros_like(all_frs_l[idx][0])
    for fr, fr_std in zip(all_frs_l[idx], all_frs_l_std[idx]):
        ax.plot(time, fr, 'k')
        propagated_error += fr_std ** 2
        ax.fill_between(time, fr - fr_std, fr + fr_std, color='k', alpha=0.25)

    fr_mean = np.mean(all_frs_l[idx], axis=0)
    fr_std = np.std(all_frs_l[idx], axis=0)
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
    sns.despine(trim=True, ax=ax)

    if save:
        plt.savefig(fig_path.joinpath('figure4_example_subject.png'))


def plot_panel_all_subjects(max_neurons, min_neurons, ax=None, save=True, plotted_regions=BRAIN_REGIONS):
    # Code to plot figure similar to figure 4c
    df = load_dataframe()
    data = load_data(event='move', norm='subtract', smoothing='kernel')

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
            frs_subj = all_frs_l[subj_idx, :]
            if df_subj.iloc[0]['institute'] not in all_present_labs:
                all_present_labs.append(df_subj.iloc[0]['institute'])
            ax[iR].plot(data['time'], np.mean(frs_subj, axis=0), c=lab_colors[df_subj.iloc[0]['institute']],
                        lw=min_lw + ((subj_idx.shape[0] - min_neurons) / (max_neurons - min_neurons)) * max_lw,
                        alpha=0.8)
        ax[iR].set_ylim(bottom=-9, top=11)
        ax[iR].axvline(0, color='k', ls='--')
        ax[iR].spines["right"].set_visible(False)
        ax[iR].spines["top"].set_visible(False)
        ax[iR].set_xlim(left=data['time'][0], right=data['time'][-1])
        # ax[iR].set_xticks([-0.2, 0, 0.2])
        sns.despine(trim=True, ax=ax[iR])
        if iR >= 1:
            ax[iR].set_yticklabels([])
        else:
            if len(plotted_regions) != 1:
                ax[iR].set_ylabel("Baselined firing rate (sp/s)")
                ax[iR].set_xlabel("Time (s)")
        ax[iR].set_title(reg)

        if iR == len(plotted_regions) - 1 and len(plotted_regions) != 1:
            # this is a hack for the legend
            for lab in all_present_labs:
                ax[iR].plot(data['time'], np.zeros_like(data['time']) - 100, c=lab_colors[lab], label=lab)
            leg = ax[iR].legend(frameon=False, bbox_to_anchor=(1, 1), labelcolor='linecolor', handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)

    if save:
        plt.savefig(fig_path.joinpath('figure4_all_subjects.png'))


if __name__ == '__main__':
    plot_main_figure()
