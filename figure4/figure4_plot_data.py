import matplotlib.pyplot as plt
import numpy as np
import figrid as fg
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs, save_figure_path, figure_style
from figure4.figure4_load_data import load_data, load_dataframe
from figure4.figure4_plot_functions import plot_raster_and_psth

lab_number_map, institution_map, lab_colors = labs()
fig_path = save_figure_path(figure='figure4')

def plot_panel_single_neuron(ax=None, save=True):
    # Code to plot figure similar to figure 4a
    pid = 'f26a6ab1-7e37-4f8d-bb50-295c056e1062'
    neuron = 386
    align_event = 'move'
    params = {'smoothing': 'sliding',
              'fr_bin_size': 0.04,
              'event_epoch': [-0.3, 0.2],
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
    idx = df_filt.loc[(df['region'] == region) & (df['subject'] == subject)].index
    time = data['time']
    for fr, fr_std in zip(all_frs_l[idx], all_frs_l_std[idx]):
        ax.plot(time, fr, 'k')
        # ax.fill_between(time, fr-fr_std, fr+fr_std, 'k', alpha=0.5)

    fr_mean = np.mean(all_frs_l[idx], axis=0)
    fr_std = np.std(all_frs_l[idx], axis=0)
    ax.plot(time, fr_mean, 'g')

    if save:
        plt.savefig(fig_path.joinpath('figure4_example_subject.png'))


def plot_panel_all_subjects(ax=None, save=True):
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
        fig, ax = plt.subplots(1, len(BRAIN_REGIONS))
    df_filt_reg = df_filt.groupby('region')
    for iR, reg in enumerate(BRAIN_REGIONS):
        df_reg = df_filt_reg.get_group(reg)
        df_reg_subj = df_reg.groupby('subject')
        for subj in df_reg_subj.groups.keys():
            df_subj = df_reg_subj.get_group(subj)
            subj_idx = df_reg_subj.groups[subj]
            frs_subj = all_frs_l[subj_idx, :]
            ax[iR].plot(np.mean(frs_subj, axis=0), c=lab_colors[df_subj.iloc[0]['institute']])

    if save:
        plt.savefig(fig_path.joinpath('figure4_all_subjects.png'))


DPI = 400  # if the figure is too big on your screen, lower this number
if __name__ == '__main__':
    figure_style()
    fig = plt.figure(figsize=(7, 10.5), dpi=DPI)  # full width figure is 7 inches
    ax = {'panel_A_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.325], yspan=[0., 0.15],
                                             wspace=0.3),
          'panel_A_2': fg.place_axes_on_grid(fig, xspan=[0.075, 0.325], yspan=[0.15, 0.3],
                                             wspace=0.3),
          'panel_B': fg.place_axes_on_grid(fig, xspan=[0.43, 1.], yspan=[0., 0.3],
                                           wspace=0.3),
          'panel_C_1': fg.place_axes_on_grid(fig, xspan=[0.075, 0.25], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_C_2': fg.place_axes_on_grid(fig, xspan=[0.26, 0.435], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_C_3': fg.place_axes_on_grid(fig, xspan=[0.445, 0.62], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_C_4': fg.place_axes_on_grid(fig, xspan=[0.63, 0.805], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_C_5': fg.place_axes_on_grid(fig, xspan=[0.815, 1.], yspan=[0.38, 0.65],
                                             wspace=0.3),
          'panel_D': fg.place_axes_on_grid(fig, xspan=[0.075, 1.], yspan=[0.7, 1],
                                           wspace=0.3)}
    plot_panel_single_neuron(ax=[ax['panel_A_1'], ax['panel_A_2']], save=False)
    plot_panel_single_subject(ax=ax['panel_B'], save=False)
    plot_panel_all_subjects(ax=[ax['panel_C_1'], ax['panel_C_2'], ax['panel_C_3'], ax['panel_C_4'], ax['panel_C_5']], save=False)
    plt.savefig(fig_path.joinpath(f'figure4_combined.png'))
    plt.show()
