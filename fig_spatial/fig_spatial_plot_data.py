import figrid as fg
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import numpy as np

from reproducible_ephys_functions import (filter_recordings, BRAIN_REGIONS, REGION_RENAME, figure_style,
                                          save_figure_path, get_row_coord, get_label_pos)
from fig_spatial.fig_spatial_load_data import load_data, load_dataframe, load_regions

PRINT_INFO = False
fig_save_path = save_figure_path(figure='fig_spatial')

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429],
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048,
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381],
 [0.0589714286, 0.6837571429, 0.7253857143],
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048],
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905],
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
 [0.7184095238, 0.7411333333, 0.3904761905],
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619],
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333],
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


def plot_fr_ff():
    # Figure 6 supplement 4
    df = load_dataframe()
    data = load_data(event='move', smoothing='sliding', norm=None)

    df_filt = filter_recordings(df, min_regions=0, min_rec_lab=0, min_lab_region=0)
    all_frs_l = data['all_frs_l'][df_filt['include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['include'] == 1]
    all_ffs_l = data['all_ffs_l'][df_filt['include'] == 1]
    all_ffs_r = data['all_ffs_r'][df_filt['include'] == 1]
    df_filt = df_filt[df_filt['include'] == 1].reset_index()

    df_filt_reg = df_filt.groupby('region')
    reg_cent_of_mass = load_regions()
    
    # For setting y limits for the firing rate each brain region; May need to change:
    ylow = [3.35, 5.8, 5.6, 5.35, 6.3]
    yhigh = [7.4, 13.1, 13.2, 9, 11]
 
    # For setting y limits for the Fano Factor each brain region; May need to change:
    ylow2 = [1.24, 1.35, 1.39, 1.53, 1.46]
    yhigh2 = [1.45, 1.9, 1.81, 1.76, 1.7]

    figure_style()
    width = 7
    height = 9
    fig = plt.figure(figsize=(width, height))

    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    gs2 = fig.add_gridspec(3, 2, wspace=0.1, hspace=0)

    ax_3d = {
        'PPC': fig.add_subplot(gs2[0, 0], projection='3d'),
        'CA1': fig.add_subplot(gs2[0, 1], projection='3d'),
        'DG': fig.add_subplot(gs2[1, 0], projection='3d'),
        'LP': fig.add_subplot(gs2[1, 1], projection='3d'),
        'PO': fig.add_subplot(gs2[2, 1], projection='3d'),
    }
    gs1 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.3)
    for regIdx, (reg, alph) in enumerate(zip(BRAIN_REGIONS, ['a', 'b', 'c', 'd', 'e'])):
        reg_title = REGION_RENAME[reg]
        gs_reg = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[regIdx, 0], wspace=0.1, hspace=0.15)
        axes = {'A': fig.add_subplot(gs_reg[0, 0]),
                'B': fig.add_subplot(gs_reg[0, 1]),
                'C': fig.add_subplot(gs_reg[1, 0]),
                'D': fig.add_subplot(gs_reg[1, 1]),
                }

        df_reg = df_filt_reg.get_group(reg)

        if PRINT_INFO:
            print(f'Figure 6 supp 4 {reg}')
            print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
                  f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')

        cofm_reg = reg_cent_of_mass[reg_cent_of_mass['region'] == reg]
        plot_fanofactor_3D(df_reg, cofm_reg, ax=ax_3d[reg], cb=True)

        reg_idx = df_filt_reg.groups[reg]

        ffs_r_reg = all_ffs_r[reg_idx, :]
        n_neurons = ffs_r_reg.shape[0]
        ffs_r_mean = np.nanmean(ffs_r_reg, axis=0)
        ffs_r_std = np.nanstd(ffs_r_reg, axis=0) / np.sqrt(n_neurons)

        frs_r_reg = all_frs_r[reg_idx, :]
        frs_r_mean = np.nanmean(frs_r_reg, axis=0)
        frs_r_std = np.nanstd(frs_r_reg, axis=0) / np.sqrt(n_neurons)

        ffs_l_reg = all_ffs_l[reg_idx, :]
        ffs_l_mean = np.nanmean(ffs_l_reg, axis=0)
        ffs_l_std = np.nanstd(ffs_l_reg, axis=0) / np.sqrt(n_neurons)

        frs_l_reg = all_frs_l[reg_idx, :]
        frs_l_mean = np.nanmean(frs_l_reg, axis=0)
        frs_l_std = np.nanstd(frs_l_reg, axis=0) / np.sqrt(n_neurons)

        ax = axes['A']
        ax.fill_between(data['time_fr'], frs_l_mean - frs_l_std, frs_l_mean + frs_l_std, color='k', alpha=0.25)
        ax.errorbar(data['time_fr'], frs_l_mean, frs_l_std, color='k', capsize=0.5, linewidth=0.5, elinewidth=0.25)
        ax.set_ylabel('Firing Rate \n (spikes/s)')
        ax.set_xticklabels([])
        if regIdx == 0:
            ax.set_title('CW Movement Post Left Stim', pad=20)
        ax.set_ylim(ylow[regIdx], yhigh[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')
        ax.set_xlim(-0.5, 1)
        ax.text(-0.35, 1.1, alph, transform=ax.transAxes,
                         fontsize=10, va='bottom',
                         ha='right', weight='bold')

        ax = axes['C']
        ax.fill_between(data['time_ff'], ffs_l_mean - ffs_l_std, ffs_l_mean + ffs_l_std, color='k', alpha=0.25)
        ax.errorbar(data['time_ff'], ffs_l_mean, ffs_l_std, color='k', capsize=0.5, linewidth=0.5, elinewidth=0.25)
        ax.set_ylabel('Fano Factor')
        if regIdx == 4:
            ax.set_xlabel('Time from movement onset (s)')
        ax.set_ylim(ylow2[regIdx], yhigh2[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')
        ax.set_xlim(-0.5, 1)

        ax = axes['B']
        ax.fill_between(data['time_fr'], frs_r_mean - frs_r_std, frs_r_mean + frs_r_std, color='k', alpha=0.25)
        ax.errorbar(data['time_fr'], frs_r_mean, frs_r_std, color='k', capsize=0.5, linewidth=0.5, elinewidth=0.25)
        if regIdx == 0:
            ax.set_title('CCW Movement Post Right Stim', pad=20)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(ylow[regIdx], yhigh[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')
        ax.set_xlim(-0.5, 1)
        ax.text(1, 0.83, reg_title, transform=ax.transAxes, fontsize=8)

        ax = axes['D']
        ax.fill_between(data['time_ff'], ffs_r_mean - ffs_r_std, ffs_r_mean + ffs_r_std, color='k', alpha=0.25, linewidth=0)
        ax.errorbar(data['time_ff'], ffs_r_mean, ffs_r_std, color='k', capsize=0.5, linewidth=0.5, elinewidth=0.25)
        if regIdx == 4:
            ax.set_xlabel('Time from movement onset (s)')
        ax.set_yticklabels([])
        ax.set_ylim(ylow2[regIdx], yhigh2[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')
        ax.set_xlim(-0.5, 1)
        rect = patches.Rectangle((40/1e3, 0), (200-40)/1e3, ax.get_ylim()[1], facecolor='green', alpha=0.2, transform=ax.transData)
        ax.add_patch(rect)


    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.98)

    # Reduce slightly the size of the 3D plots
    for reg in BRAIN_REGIONS:
        pos = ax_3d[reg].get_position()
        pos.y1 = pos.y1 - 0.05
        pos.x1 = pos.x1 - 0.05
        ax_3d[reg].set_position(pos)

    # Then move the plots (yikes)
    for reg, sh in zip(['PPC', 'CA1', 'DG',  'LP', 'PO'], [0.11, -0.08, 0.03, -0.16, -0.05]):
        pos = ax_3d[reg].get_position()
        pos.y0 = pos.y0 + sh
        pos.y1 = pos.y1 + sh
        ax_3d[reg].set_position(pos)

    x0 = pos.x0
    x1 = pos.x1

    for reg in ['PPC', 'DG']:
        pos = ax_3d[reg].get_position()
        pos.x0 = x0
        pos.x1 = x1
        ax_3d[reg].set_position(pos)

    # Sort out the ordering so things appear nicely
    for i, reg in enumerate(BRAIN_REGIONS):
        ax_3d[reg].set_zorder(i)
        ax_3d[reg].patch.set_alpha(0.0)

    fig.savefig(fig_save_path.joinpath(f'fig_spatial_FF.pdf'))
    fig.savefig(fig_save_path.joinpath(f'fig_spatial_FF.svg'))
    plt.close()


def plot_waveforms():
    # Figure 6 supplement 5
    thresh = 0.35
    df = load_dataframe()
    df_filt = filter_recordings(df, min_channels_region=5, min_neuron_region=4, min_regions=0, min_rec_lab=0,
                                min_lab_region=0)
    df_filt = df_filt[df_filt['include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby('region')

    figure_style()
    width = 7
    height = 7
    fig = plt.figure(figsize=(width, height))

    xspans = get_row_coord(width, [1, 1], pad=0.6)
    xwidth = xspans[1][1] - xspans[1][0]
    xspans_row3 = get_row_coord(width, [1], span=[(1 - xwidth) / 2, (1 - xwidth) / 2 + xwidth], pad=0)
    yspans = get_row_coord(height, [1, 1, 1], hspace=1, pad=0.3)

    axes = {'PPC': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=[2, 1], hspace=0.8),
          'CA1': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0], dim=[2, 1], hspace=0.8),
          'DG': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1], dim=[2, 1], hspace=0.8),
          'LP': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1], dim=[2, 1], hspace=0.8),
          'PO': fg.place_axes_on_grid(fig, xspan=xspans_row3[0], yspan=yspans[2], dim=[2, 1], hspace=0.8),
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width,xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspans[1][0], pad=0.4),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width, xspans[1][0], pad=0.4),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width, xspans_row3[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'fontweight': 'bold', 'ha': 'right', 'va': 'bottom'},
              ]
    fg.add_labels(fig, labels)

    for i, reg in enumerate(BRAIN_REGIONS):
        ax = axes[reg]
        reg_title = REGION_RENAME[reg]
        df_reg = df_filt_reg.get_group(reg)

        if PRINT_INFO:
            print(f'Figure 6 supp 5 {reg}')
            print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
                  f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')

        binwidth = 0.04
        vals = df_reg.p2t.values
        minmax = (np.min(vals), np.max(vals))
        nbins = int((minmax[1] - minmax[0]) / binwidth)


        hist_amp, bins_amp = np.histogram(vals, bins=nbins, range=minmax)
        hist_amp = hist_amp / vals.size
        x_center = (bins_amp[:-1] + bins_amp[1:]) / 2
        ax[0].bar(x_center, hist_amp, width=binwidth, color='royalblue', edgecolor='k')
        ax[0].set_xlabel('Spike width (ms)')
        if np.mod(i, 2) == 0:
            ax[0].set_ylabel('Probability \n density')
        ax[0].set_title(reg_title)
        ylim = ax[0].get_ylim()
        rect = patches.Rectangle((0, 0), thresh, ylim[1], facecolor=[0.5, 0.5, 0.5, 0.2],
                                 edgecolor=[0.5, 0.5, 0.5], lw=1.5,
                                 transform=ax[0].transData)
        ax[0].add_patch(rect)
        rect = patches.Rectangle((thresh+0.005, 0), np.max(vals) - thresh, ylim[1], facecolor=[0, 0.7, 0.7, 0.2],
                                 edgecolor=[0, 0.7, 0.7], lw=1.5,
                                 transform=ax[0].transData)

        ax[0].add_patch(rect)
        ax[0].set_xlim(-1, 1.3)

        from statsmodels.distributions.empirical_distribution import ECDF
        width1 = [np.nanmin(vals), 0]
        avg_fr1 = df_reg.avg_fr[np.bitwise_and(vals >= width1[0], vals <= width1[1])]
        width2 = [0, thresh]
        avg_fr2 = df_reg.avg_fr[np.bitwise_and(vals > width2[0], vals <= width2[1])]
        width3 = [thresh, np.max(vals)]
        avg_fr3 = df_reg.avg_fr[np.bitwise_and(vals > width3[0], vals <= width3[1])]

        ecdf1 = ECDF(avg_fr1)
        ecdf2 = ECDF(avg_fr2)
        ecdf3 = ECDF(avg_fr3)

        ax[1].plot(ecdf1.x, ecdf1.y, c=[0.5, 0.5, 0.5], linestyle='-.', label='Negative spike width')
        ax[1].plot(ecdf2.x, ecdf2.y, c='k', label='Narrow spike width')
        ax[1].plot(ecdf3.x, ecdf3.y, c=[0, 0.7, 0.7], label='Wide spike width')
        ax[1].set_xlabel('Average firing rate (spikes/s)')
        if np.mod(i, 2) == 0:
            ax[1].set_ylabel('Cumulative \n probability')
        if i == 0:
            ax[1].legend(loc=4, frameon=False, fontsize=6)


    adjust = 0.3
    fig.subplots_adjust(top=1 - adjust / height, bottom=adjust / height, left=(adjust) / width,
                        right=1 - adjust / width)

    plt.savefig(fig_save_path.joinpath(f'fig_wfs.pdf'))
    plt.savefig(fig_save_path.joinpath(f'fig_wfs.svg'))
    plt.close()


def plot_task_modulation(condition0, condition1, modulated, binwidth=0.2,
                         xlabel=None, ylabel=None, leg=True, title=None, cols=['orange', 'grey'], ax=None):

    # Split the firing rates for the two conditions according to the neurons that
    # were modulated by the task
    fr0_tm = condition0[modulated]
    fr0_nm = condition0[~modulated]
    fr1_tm = condition1[modulated]
    fr1_nm = condition1[~modulated]

    delta_tm = fr1_tm - fr0_tm
    delta_nm = fr1_nm - fr0_nm

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()

    # Plot histogram for task modulated neurons
    minmax_tm = (np.min(delta_tm), np.max(delta_tm))
    nbins_tm = int((minmax_tm[1] - minmax_tm[0]) / binwidth)
    ax.hist(delta_tm, range=minmax_tm, bins=nbins_tm, histtype='step', color=cols[0])

    # Plot histogram for non task modulated neurons
    minmax_nm = (np.min(delta_nm), np.max(delta_nm))
    nbins_nm = int((minmax_nm[1] - minmax_nm[0]) / binwidth)
    ax.hist(delta_nm, range=minmax_nm, bins=nbins_nm, histtype='step', color=cols[1])

    if leg:
        legend = ax.legend(['Task modulated', 'Not modulated'], fontsize=7, frameon=False, loc='center right',
                           bbox_to_anchor=(1.25, 0.5))
        for handle in legend.legend_handles:
            handle.set_visible(False)
        for text, col in zip(legend.get_texts(), cols):
            text.set_color(col)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig, ax


def plot_task_modulation_3D(df_reg, cofm_reg, tm_test='pre_move', hists={'x': 3e2, 'y': 7e2, 'z': 6.5e2},
                            camera=(13, -111), cols=['orange', 'grey'], ax=None):

    x = (df_reg['x'].values - cofm_reg['x'].values) * 1e6
    y = (df_reg['y'].values - cofm_reg['y'].values) * 1e6
    z = (df_reg['z'].values - cofm_reg['z'].values) * 1e6

    # Generate jitter and center it
    jitt_size = np.ones(3) * min([np.ptp(np.abs(x)),np.ptp(np.abs(y)), np.ptp(np.abs(x))]) / 70
    jitt = jitt_size * (np.random.rand(x.shape[0], 3) - 0.5)

    x_jitt = x + jitt[:, 0]
    y_jitt = y + jitt[:, 1]
    z_jitt = z + jitt[:, 2]

    tm = df_reg[tm_test]
    nm = ~tm

    ax.scatter(x_jitt[tm], y_jitt[tm], z_jitt[tm], c=cols[0], marker='o', s=10, edgecolor='w', alpha=0.5)
    ax.scatter(x_jitt[nm], y_jitt[nm], z_jitt[nm], c=cols[1], marker='o', s=0.4, alpha=0.8, zorder=5)

    # Plot the centre of mass
    ax.scatter(0, 0, 0, c='r', marker='x', s=40, linewidths=3, zorder=10)

    if 'x' in hists.keys():
        plot_histograms_along_x_axis(x, tm, nm, np.max(y), np.max(z) + 10, sig='**', scale=hists['x'], cols=cols, ax=ax)
    if 'y' in hists.keys():
        plot_histograms_along_y_axis(y, tm, nm, np.max(x), np.max(z) + 10, sig='**', scale=hists['y'], cols=cols, ax=ax)
    if 'z' in hists.keys():
        plot_histograms_along_z_axis(z, tm, nm, np.max(x), np.min(y) - 10, sig='**', scale=hists['z'], cols=cols,ax=ax)

    ax.set_xlabel('\u0394ML (\u03bcm)', labelpad=-10)
    ax.set_ylabel('\u0394AP (\u03bcm)', labelpad=-7)
    ax.set_zlabel('\u0394DV (\u03bcm)', labelpad=-8)
    ax.tick_params(axis='x', pad=-5)
    ax.tick_params(axis='y', pad=-1)
    ax.tick_params(axis='z', pad=-2)
    ax.xaxis.set_major_locator(MultipleLocator(300))
    ax.yaxis.set_major_locator(MultipleLocator(300))
    ax.zaxis.set_major_locator(MultipleLocator(200))
    for label in ax.get_yticklabels():
        label.set_verticalalignment('bottom')
    ax.view_init(elev=camera[0], azim=camera[1])
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('darkgray')
    ax.yaxis.pane.set_edgecolor('darkgray')
    ax.zaxis.pane.set_edgecolor('darkgray')
    ax.grid(False)

    return tm, nm


def plot_main_figure():
    # Figure 6 and supp 1 and 2
    df = load_dataframe()
    df_filt = filter_recordings(df, min_channels_region=5, min_neuron_region=4, min_regions=0, min_rec_lab=0,
                                min_lab_region=0)
    df_filt = df_filt[df_filt['include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby('region')
    reg_cent_of_mass = load_regions()

    scales = {
        'LP': {'fr': {'x': 3e2, 'y': 7e2}, 'tm': {'z': 4.5e2}, 'tm_lr': {'z': 4.5e2}},
        'PPC': {'fr': {'x': 5e2}, 'tm': {'z': 3e2}, 'tm_lr': {}},
        'CA1': {'fr': {'z': 3e2}, 'tm': {}, 'tm_lr': {'z': 3e2}},
    }

    for reg in ['LP', 'PPC', 'CA1']:

        figure_style()
        width = 7
        height = 7
        fig = plt.figure(figsize=(width, height))

        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[4, 5], wspace=0.1, hspace=0.1)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[1, 5], hspace=0)
        gs11 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs1[0], wspace=0)
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[1, 5], hspace=0)
        gs21 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs2[0], wspace=0)

        axes = {'A': fig.add_subplot(gs[0, 0]),
                'B': fig.add_subplot(gs[0, 1], projection='3d'),
                'C_1': fig.add_subplot(gs11[1:-1]),
                'D_1': fig.add_subplot(gs21[1:-1]),
                'C_2': fig.add_subplot(gs1[1], projection='3d'),
                'D_2': fig.add_subplot(gs2[1], projection='3d'),
                }

        axes['C_1'].text(-0.35, 6, 'a',
                         transform=axes['C_1'].transAxes,
                         fontsize=10, va='bottom',
                         ha='right', weight='bold')
        axes['D_1'].text(-0.7, 6 ,'b',
                         transform=axes['D_1'].transAxes,
                         fontsize=10, va='bottom',
                         ha='right', weight='bold')
        axes['C_1'].text(-0.35, 1.30, 'c',
                         transform=axes['C_1'].transAxes,
                         fontsize=10, va='bottom',
                         ha='right', weight='bold')
        axes['D_1'].text(-0.35, 1.30, 'd',
                         transform=axes['D_1'].transAxes,
                         fontsize=10, va='bottom',
                         ha='right', weight='bold')

        df_reg = df_filt_reg.get_group(reg)

        if PRINT_INFO:
            print(f'Figure 6 {reg}')
            print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
                  f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')

        cofm_reg = reg_cent_of_mass[reg_cent_of_mass['region'] == reg]

        axes['A'].set_axis_off()

        # Firing rate 3D plot
        cond1, cond2 = plot_firing_rate_3D(df_reg, cofm_reg, reg=reg, hists=scales[reg]['fr'], inset=False, ax=axes['B'])
        if reg == 'LP':
            inset = axes['B'].inset_axes([0.82, 0.5, 0.25, 0.1])
            plot_inset_histogram(df_reg.amp * 1e6, cond1, cond2, bins=30, xlabel='Amplitude (\u03bcv)', ylabel='Probability',
                                 cols=['tab:orange', 'tab:blue'], sig='**', sig_loc=[0.3, 0.5], ax=inset,
                                 minmax=(0, np.max(df_reg.amp * 1e6)))


        # Pre move test
        plot_task_modulation_3D(df_reg, cofm_reg, tm_test='pre_move', hists=scales[reg]['tm'], ax=axes['C_2'])
        plot_task_modulation(df_reg.avg_fr_base, df_reg.avg_fr_pre_move, df_reg.pre_move,
                                 xlabel=r'Firing rate change (early movement - pre stim)', ylabel='Number of neurons',
                                 title='Movement initiation vs pre-stim test', leg=True, ax=axes['C_1'])

        axes['C_1'].set_zorder(2)
        axes['C_2'].set_zorder(1)

        # Left right test
        cond1, cond2 = plot_task_modulation_3D(df_reg, cofm_reg, tm_test='pre_move_lr', hists=scales[reg]['tm_lr'], ax=axes['D_2'])
        plot_task_modulation(df_reg.avg_fr_pre_moveL, df_reg.avg_fr_pre_moveR, df_reg.pre_move_lr,
                             xlabel=r'Firing rate change (right stim - left stim)', ylabel='Number of neurons',
                             title='Left vs right movement test', leg=True, ax=axes['D_1'])
        if reg == 'LP':
            inset = axes['D_2'].inset_axes([0.55, 0.05, 0.25, 0.1])
            plot_inset_histogram(df_reg.amp * 1e6, cond1, cond2, bins=30, xlabel='Amplitude (\u03bcv)', ylabel='Probability',
                                 cols=['orange', 'grey'], sig='**', sig_loc=[0.3, 0.5], ax=inset,
                                 minmax=(0, np.max(df_reg.amp * 1e6)))

            inset = axes['D_2'].inset_axes([0.88, 0.05, 0.2, 0.1])
            plot_inset_histogram(df_reg.p2t, cond1, cond2, bins=15, xlabel='Waveform duration (ms)', ylabel='',
                                 cols=['orange', 'grey'], sig='*', sig_loc=[0.4, 0.5], ax=inset)

            axes['D_2'].set_xlabel('')
            axes['D_2'].set_xticklabels([])

        axes['D_1'].set_zorder(2)
        axes['D_2'].set_zorder(1)

        # adjust = 0.3
        # if reg == 'LP':
        #     fig.subplots_adjust(top=1 - (adjust-0.2) / height, bottom=(adjust) / height, left=(adjust) / width,
        #                         right=1 - (adjust + 0.1) / width)
        # else:
        #     fig.subplots_adjust(top=1 - (adjust-0.2) / height, bottom=(adjust - 0.2) / height, left=(adjust) / width,
        #                         right=1 - (adjust - 0.2) / width)

        if reg == 'LP':
            fig.subplots_adjust(right=0.94, bottom=1-0.97, top=0.98, left=0.03)
        else:
            fig.subplots_adjust(right=0.98, bottom=0, top=0.98, left=0.04)

        plt.savefig(fig_save_path.joinpath(f'fig_{reg}.pdf'))
        plt.savefig(fig_save_path.joinpath(f'fig_{reg}.svg'))
        plt.close()


def plot_supp1():
    # Figure 6 supp 3
    df = load_dataframe()
    df_filt = filter_recordings(df, min_channels_region=5, min_neuron_region=4, min_regions=0, min_rec_lab=0,
                                min_lab_region=0)
    df_filt = df_filt[df_filt['include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby('region')
    reg_cent_of_mass = load_regions()

    scales = {
        'LP': {'fr': {'x': 3e2, 'y': 7e2}, 'tm': {'z': 5.8e2}, 'tm_lr': {'z': 6.5e2}},
        'PPC': {'fr': {'x': 5e2}, 'tm': {'z': 4.3e2}, 'tm_lr': {}},
        'CA1': {'fr': {'z': 5e2}, 'tm': {}, 'tm_lr': {'z': 5e2}},
        'DG': {'fr': {}, 'tm': {}},
        'PO': {'fr': {'y': 6e2}, 'tm': {}}
    }

    figure_style()
    width = 7
    height = 6
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 2, wspace=0.1, hspace=0)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], height_ratios=[1, 25], hspace=-0.1)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], height_ratios=[1, 25], hspace=-0.05)
    gs3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[1, 25], hspace=-0.05)
    gs4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[1, 25], hspace=-0.05)

    axes = {'A_1': fig.add_subplot(gs1[0]),
            'A_2': fig.add_subplot(gs1[1], projection='3d'),
            'B_1': fig.add_subplot(gs2[0]),
            'B_2': fig.add_subplot(gs2[1], projection='3d'),
            'C_1': fig.add_subplot(gs3[0]),
            'C_2': fig.add_subplot(gs3[1], projection='3d'),
            'D_1': fig.add_subplot(gs4[0]),
            'D_2': fig.add_subplot(gs4[1], projection='3d'),
            }
    reg = 'DG'
    df_reg = df_filt_reg.get_group(reg)

    if PRINT_INFO:
        print(f'Figure 6 {reg}')
        print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
              f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')

    cofm_reg = reg_cent_of_mass[reg_cent_of_mass['region'] == reg]

    axes['A_1'].set_axis_off()
    axes['A_1'].text(0.4, -2, reg,
                     transform=axes['A_1'].transAxes,
                     fontsize=mpl.rcParams["axes.titlesize"], va='bottom',
                     ha='center')
    axes['A_1'].text(-0.25, -0.5, 'a',
                     transform=axes['A_1'].transAxes,
                     fontsize=10, va='bottom',
                     ha='right', weight='bold')

    cond1, cond2 = plot_firing_rate_3D(df_reg, cofm_reg, reg=reg, hists=scales[reg]['fr'], inset=False, ax=axes['A_2'])
    inset = axes['A_2'].inset_axes([0.84, 0.5, 0.2, 0.1])
    plot_inset_histogram(df_reg.p2t, cond1, cond2, bins=15, xlabel='Waveform duration (ms)', ylabel='Probability',
                         cols=['tab:orange', 'tab:blue'], sig='*', sig_loc=[0.25, 0.5], ax=inset)

    axes['A_1'].set_zorder(2)
    axes['A_2'].set_zorder(1)


    axes['B_1'].set_axis_off()
    axes['B_1'].text(-0.025, -0.5, 'b',
                     transform=axes['B_1'].transAxes,
                     fontsize=10, va='bottom',
                     ha='right', weight='bold')
    axes['B_1'].text(0.5, -2, 'Movement initiation vs pre-stim test',
                     transform=axes['B_1'].transAxes,
                     fontsize=mpl.rcParams["axes.titlesize"], va='bottom',
                     ha='center')

    cond1, cond2 = plot_task_modulation_3D(df_reg, cofm_reg, tm_test='pre_move', hists=scales[reg]['tm'], ax=axes['B_2'])
    inset = axes['B_2'].inset_axes([0.84, 0.5, 0.2, 0.1])
    plot_inset_histogram(df_reg.p2t, cond1, cond2, bins=15, xlabel='Waveform duration (ms)', ylabel='Probability',
                         cols=['orange', 'grey'], sig='*', sig_loc=[0.25, 0.5], ax=inset)

    axes['B_1'].set_zorder(2)
    axes['B_2'].set_zorder(1)

    reg = 'PO'
    df_reg = df_filt_reg.get_group(reg)

    if PRINT_INFO:
        print(f'Figure 6 {reg}')
        print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
              f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')

    cofm_reg = reg_cent_of_mass[reg_cent_of_mass['region'] == reg]


    axes['C_1'].set_axis_off()
    axes['C_1'].text(0.4, -2, reg,
                     transform=axes['C_1'].transAxes,
                     fontsize=mpl.rcParams["axes.titlesize"], va='bottom',
                     ha='center')
    axes['C_1'].text(-0.25, -0.5, 'c',
                     transform=axes['C_1'].transAxes,
                     fontsize=10, va='bottom',
                     ha='right', weight='bold')

    cond1, cond2 = plot_firing_rate_3D(df_reg, cofm_reg, reg=reg, hists=scales[reg]['fr'], inset=False, ax=axes['C_2'])
    inset = axes['C_2'].inset_axes([0.35, -0.05, 0.25, 0.1])
    plot_inset_histogram(df_reg.amp * 1e6, cond1, cond2, bins=30, xlabel='Amplitude (\u03bcv)', ylabel='Probability',
                         cols=['orange', 'grey'], sig='**', sig_loc=[0.3, 0.5], ax=inset,
                         minmax=(0, np.max(df_reg.amp * 1e6)))

    inset = axes['C_2'].inset_axes([0.7, -0.05, 0.2, 0.1])
    plot_inset_histogram(df_reg.p2t, cond1, cond2, bins=15, xlabel='Waveform duration (ms)', ylabel='',
                         cols=['orange', 'grey'], sig='**', sig_loc=[0.3, 0.5], ax=inset)

    axes['C_1'].set_zorder(2)
    axes['C_2'].set_zorder(1)

    axes['D_1'].set_axis_off()
    axes['D_1'].text(-0.025, -0.5, 'd',
                     transform=axes['D_1'].transAxes,
                     fontsize=10, va='bottom',
                     ha='right', weight='bold')
    axes['D_1'].text(0.5, -2, 'Movement initiation vs pre-stim test',
                     transform=axes['D_1'].transAxes,
                     fontsize=mpl.rcParams["axes.titlesize"], va='bottom',
                     ha='center')

    cond1, cond2 = plot_task_modulation_3D(df_reg, cofm_reg, tm_test='pre_move', hists=scales[reg]['tm'], ax=axes['D_2'])

    axes['D_1'].set_zorder(2)
    axes['D_2'].set_zorder(1)

    fig.subplots_adjust(right=0.96, bottom=0.08, top=0.99, left=0.12)

    plt.savefig(fig_save_path.joinpath(f'fig_PO_DG.pdf'))
    plt.savefig(fig_save_path.joinpath(f'fig_PO_DG.svg'))
    plt.close()


def plot_inset_histogram(vals, condition1, condition2, bins=30, xlabel=None, ylabel=None, minmax=None,
                         cols=['orange', 'blue'], sig='**', sig_loc=[0, 0], ax=None):

    if minmax is None:
        minmax = (np.min(vals), np.max(vals))
    hist_amp, bins_amp = np.histogram(vals[condition1], bins=bins, range=minmax)
    hist_amp = hist_amp / vals[condition1].size
    x_center = (bins_amp[:-1] + bins_amp[1:]) / 2
    ax.step(x_center, hist_amp, color=cols[0])

    hist_amp, bins_amp = np.histogram(vals[condition2], bins=bins, range=(np.min(vals), np.max(vals)))
    hist_amp = hist_amp / vals[condition2].size
    x_center = (bins_amp[:-1] + bins_amp[1:]) / 2
    ax.step(x_center, hist_amp, color=cols[1])

    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.text(sig_loc[0], sig_loc[1], sig, fontsize=9, transform=ax.transAxes)


def compute_hist_and_percentiles(vals, incl, binwidth=100):

    minmax = (np.min(vals) - binwidth, np.max(vals) + binwidth)
    nbins = int((minmax[1] - minmax[0]) / binwidth) + 1

    # Compute histogram and percentiles for condition 1
    hist, bins = np.histogram(vals[incl], bins=nbins, range=minmax)
    hist = hist / vals[incl].size
    bins = (bins[:-1] + bins[1:]) / 2

    pc_20 = np.percentile(vals[incl], 20)
    pc_80 = np.percentile(vals[incl], 80)

    return hist, bins, pc_20, pc_80


def plot_histograms_along_x_axis(vals, condition1, condition2, ymax, zmax, scale=5e2, binwidth=100,
                                cols=['tab:orange', 'tab:blue'], sig='', ax=None):

    # Compute histogram and percentiles for two conditions
    hist_1, bins_1, pc_20_1, pc_80_1 = compute_hist_and_percentiles(vals, condition1, binwidth=binwidth)
    hist_2, bins_2, pc_20_2, pc_80_2 = compute_hist_and_percentiles(vals, condition2, binwidth=binwidth)

    # Compute the amount we scale the plots by
    bar_max = np.max([np.max(hist_1), np.max(hist_2)])
    width_max = np.max([pc_20_1, pc_80_1])

    # Add plots to figure
    # Condition 1
    ax.bar([(pc_20_1 + pc_80_1) / 2], [bar_max * scale * 1.1], zs=ymax, zdir='y', alpha=0.2, width=np.abs(pc_20_1 - pc_80_1), color=cols[0], bottom=zmax,
           zorder=2, align='center')
    ax.step(bins_1, hist_1 * scale + zmax, zs=ymax, zdir='y', color=cols[0], where='mid')

    # Condition 2
    ax.bar([(pc_20_2 + pc_80_2) / 2], [bar_max * scale * 0.9], zs=ymax, zdir='y', alpha=0.2, width=np.abs(pc_20_2 - pc_80_2), color=cols[1], bottom=zmax,
           zorder=2, align='center')
    ax.step(bins_2, hist_2 * scale + zmax, zs=ymax, zdir='y', color=cols[1], where='mid')

    if sig:
        ax.text(width_max + 180, ymax, (bar_max * scale * 1) + zmax, '**', 'x', fontsize=10, ha='right', va='center')
        ax.quiver(
            width_max + 200, ymax, (bar_max * scale * 1) + zmax,  # Start of the arrow
            -200, 0, 0,  # Direction vector (x, y, z components)
            color=cols[0], linewidth=1
        )


def plot_histograms_along_y_axis(vals, condition1, condition2, xmax, zmax, scale=5e2, binwidth=100,
                                 cols=['tab:orange', 'tab:blue'], sig='', ax=None):
    # Compute histogram and percentiles for two conditions
    hist_1, bins_1, pc_20_1, pc_80_1 = compute_hist_and_percentiles(vals, condition1, binwidth=binwidth)
    hist_2, bins_2, pc_20_2, pc_80_2 = compute_hist_and_percentiles(vals, condition2, binwidth=binwidth)

    # Compute the amount we scale the plots by
    bar_max = np.max([np.max(hist_1), np.max(hist_2)])
    width_max = np.min([pc_20_1, pc_80_1])

    # Add plots to figure
    # Condition 1
    ax.bar([(pc_20_1 + pc_80_1) / 2], [bar_max * scale * 1.1], zs=xmax, zdir='x', alpha=0.2, width=np.abs(pc_20_1 - pc_80_1), color=cols[0],
           bottom=zmax, zorder=2, align='center')
    ax.step(bins_1, hist_1 * scale + zmax, zs=xmax, zdir='x', color=cols[0], where='mid')

    # Condition 2
    ax.bar([(pc_20_2 + pc_80_2) / 2], [bar_max * scale * 0.9], zs=xmax, zdir='x', alpha=0.2, width=np.abs(pc_20_2 - pc_80_2), color=cols[1],
           bottom=zmax, zorder=2, align='center')
    ax.step(bins_2, hist_2 * scale + zmax, zs=xmax, zdir='x', color=cols[1], where='mid')

    if sig:
        ax.text(xmax, width_max - 180, (bar_max * scale * 1) + zmax, '**', 'y', fontsize=10, ha='center', va='center')
        ax.quiver(
            xmax, width_max - 300, (bar_max * scale * 1) + zmax,  # Start of the arrow
            0, 250, 0,  # Direction vector (x, y, z components)
            color=cols[0], linewidth=1
        )


def plot_histograms_along_z_axis(vals, condition1, condition2, xmax, ymax, scale=6.5e2, binwidth=100,
                                 cols=['tab:orange', 'tab:blue'], sig='', ax=None):
    # Compute histogram and percentiles for two conditions
    hist_1, bins_1, pc_20_1, pc_80_1 = compute_hist_and_percentiles(vals, condition1, binwidth=binwidth)
    hist_2, bins_2, pc_20_2, pc_80_2 = compute_hist_and_percentiles(vals, condition2, binwidth=binwidth)

    # Compute the amount we scale the plots by
    bar_max = np.max([np.max(hist_1), np.max(hist_2)])
    width_max = np.min([pc_20_1, pc_80_1])

    # Add plots to figure
    # Condition 1
    # Bottom should be the deviation from the zaxis to start the triangle
    # Height should be the width
    ax.bar([xmax], [np.abs(pc_20_1 - pc_80_1)], zs=ymax, zdir='y', alpha=0.2, width=[bar_max * scale * 1.1], color=cols[0],
           bottom=pc_20_1, zorder=2, align='edge')
    ax.step(hist_1 * scale + xmax, bins_1, zs=ymax, zdir='y', color=cols[0], where='mid')

    # Condition 2
    ax.bar([xmax], [np.abs(pc_20_2 - pc_80_2)], zs=ymax, zdir='y', alpha=0.2, width=[bar_max * scale * 0.9], color=cols[1],
           bottom=pc_20_2, zorder=2, align='edge')
    ax.step(hist_2 * scale + xmax, bins_2, zs=ymax, zdir='y', color=cols[1], where='mid')

    if sig:
        ax.text((bar_max * scale * 1.2) + xmax, ymax, width_max - 50, '**', 'z', fontsize=10, ha='left', va='center')
        ax.quiver(
            (bar_max * scale * 1.1) + xmax, ymax, width_max - 150, # Start of the arrow
            0, 0, 150,  # Direction vector (x, y, z components)
            color=cols[0], linewidth=1
        )


def plot_firing_rate_3D(df_reg, cofm_reg, reg=None, hists={'x': 3e2, 'y': 7e2}, camera=(13, -111), inset=False, ax=None):


    x = (df_reg['x'].values - cofm_reg['x'].values) * 1e6
    y = (df_reg['y'].values - cofm_reg['y'].values) * 1e6
    z = (df_reg['z'].values - cofm_reg['z'].values) * 1e6

    # Generate jitter and center it
    jitt_size = np.ones(3) * min([np.ptp(np.abs(x)),np.ptp(np.abs(y)), np.ptp(np.abs(x))]) / 70
    jitt = jitt_size * (np.random.rand(x.shape[0], 3) - 0.5)

    x_jitt = x + jitt[:, 0]
    y_jitt = y + jitt[:, 1]
    z_jitt = z + jitt[:, 2]

    avg_fr = df_reg.avg_fr.values
    log_avg_fr = np.log10(avg_fr)

    # filt_thres = 0.15
    # deviation_from_median = (avg_fr - np.median(avg_fr)) / (np.max(avg_fr) - np.min(avg_fr))
    # regular_units = np.abs(deviation_from_median) < filt_thres
    # outlier_units = np.abs(deviation_from_median) >= filt_thres
    filt_thres = 13.5
    regular_units = avg_fr < filt_thres
    outlier_units = avg_fr >= filt_thres

    norm = colors.Normalize(vmin=np.min(log_avg_fr), vmax=np.max(log_avg_fr), clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=parula_map)
    cluster_color = np.array([mapper.to_rgba(col) for col in log_avg_fr])

    ax.scatter(x_jitt[regular_units], y_jitt[regular_units], z_jitt[regular_units], c=cluster_color[regular_units], marker='o',
               s=0.4, alpha=0.5)

    ax.scatter(x_jitt[outlier_units], y_jitt[outlier_units], z_jitt[outlier_units], c=cluster_color[outlier_units], marker='o',
                      s=15, edgecolor='k', alpha=1, zorder=5)

    ax.scatter(0, 0, 0, c='r', marker='x', s=30, linewidths=3, zorder=10)

    ax_cbar = ax.inset_axes([-0.15, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=parula_map), cax=ax_cbar)
    # TODO need to make this so it is computed
    ticks = np.array([0.1, 0.3, 1, 3, 10, 30]) if reg not in ['LP', 'PO'] else np.array([0.3, 1, 3, 10, 30])
    cbar.set_ticks(np.log10(ticks))
    cbar.set_ticklabels(ticks)
    cbar.set_label('Average firing rate (spikes/s)', rotation=90, labelpad=-40)

    ax_cbar.axhline(np.log10(filt_thres), xmin=ax_cbar.get_xlim()[0] - 0.4, xmax=ax_cbar.get_xlim()[1] + 0.4, c='k',
                    clip_on=False, lw=1.5)

    if 'x' in hists.keys():
        plot_histograms_along_x_axis(x, outlier_units, regular_units, np.max(y), np.max(z) + 10, scale=hists['x'], sig='**', ax=ax)

    if 'y' in hists.keys():
        plot_histograms_along_y_axis(y, outlier_units, regular_units, np.max(x), np.max(z) + 10, scale=hists['y'], sig='**', ax=ax)

    if 'z' in hists.keys():
        plot_histograms_along_z_axis(z, outlier_units, regular_units, np.max(x), np.min(y) - 10, scale=hists['z'], sig='**', ax=ax)

    ax.set_xlabel('\u0394ML (\u03bcm)', labelpad=-10)
    ax.set_ylabel('\u0394AP (\u03bcm)', labelpad=-7)
    ax.set_zlabel('\u0394DV (\u03bcm)', labelpad=-8)
    ax.tick_params(axis='x', pad=-5)
    ax.tick_params(axis='y', pad=-1)
    ax.tick_params(axis='z', pad=-2)
    ax.xaxis.set_major_locator(MultipleLocator(300))
    ax.yaxis.set_major_locator(MultipleLocator(300))
    ax.zaxis.set_major_locator(MultipleLocator(200))
    for label in ax.get_yticklabels():
        label.set_verticalalignment('bottom')
    ax.view_init(elev=camera[0], azim=camera[1])
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('darkgray')
    ax.yaxis.pane.set_edgecolor('darkgray')
    ax.zaxis.pane.set_edgecolor('darkgray')
    ax.grid(False)

    return outlier_units, regular_units


def plot_fanofactor_3D(df_reg, cofm_reg, camera=(13, -111), cb=False, ax=None):


    x = (df_reg['x'].values - cofm_reg['x'].values) * 1e6
    y = (df_reg['y'].values - cofm_reg['y'].values) * 1e6
    z = (df_reg['z'].values - cofm_reg['z'].values) * 1e6

    # Generate jitter and center it
    jitt_size = np.ones(3) * min([np.ptp(np.abs(x)),np.ptp(np.abs(y)), np.ptp(np.abs(x))]) / 70
    jitt = jitt_size * (np.random.rand(x.shape[0], 3) - 0.5)

    x_jitt = x + jitt[:, 0]
    y_jitt = y + jitt[:, 1]
    z_jitt = z + jitt[:, 2]

    avg_ff = df_reg.avg_ff_post_move.values

    filt_thres = 1
    regular_units = avg_ff > filt_thres
    outlier_units = avg_ff <= filt_thres

    norm = colors.Normalize(vmin=0.5, vmax=3, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=parula_map)
    cluster_color = np.array([mapper.to_rgba(col) for col in avg_ff ])

    ax.scatter(x_jitt[regular_units], y_jitt[regular_units], z_jitt[regular_units], c=cluster_color[regular_units], marker='o',
               s=0.4, alpha=0.5)

    ax.scatter(x_jitt[outlier_units], y_jitt[outlier_units], z_jitt[outlier_units], c=cluster_color[outlier_units], marker='o',
               s=10, edgecolor='k', lw=0.5, alpha=0.8, zorder=5)

    ax.scatter(0, 0, 0, c='r', marker='x', s=30, linewidths=3, zorder=10)

    if cb:
        ax_cbar = ax.inset_axes([1.03, 0.15, 0.03, 0.6])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=parula_map), cax=ax_cbar)
        cbar.set_label('Fano factor', rotation=90, labelpad=-30)
        ax_cbar.axhline(filt_thres, xmin=ax_cbar.get_xlim()[0] - 0.4, xmax=ax_cbar.get_xlim()[1] + 0.4, c='k',
                        clip_on=False, lw=1.5)

    ax.set_xlabel('\u0394ML (\u03bcm)', labelpad=-10)
    ax.set_ylabel('\u0394AP (\u03bcm)', labelpad=-7)
    ax.set_zlabel('\u0394DV (\u03bcm)', labelpad=-8)
    ax.tick_params(axis='x', pad=-5)
    ax.tick_params(axis='y', pad=-1)
    ax.tick_params(axis='z', pad=-2)
    ax.xaxis.set_major_locator(MultipleLocator(300))
    ax.yaxis.set_major_locator(MultipleLocator(300))
    ax.zaxis.set_major_locator(MultipleLocator(200))
    for label in ax.get_yticklabels():
        label.set_verticalalignment('bottom')
    ax.view_init(elev=camera[0], azim=camera[1])
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('darkgray')
    ax.yaxis.pane.set_edgecolor('darkgray')
    ax.zaxis.pane.set_edgecolor('darkgray')
    ax.grid(False)


if __name__ == '__main__':
    plot_main_figure()
    plot_supp1()
    plot_waveforms()
    plot_fr_ff()
