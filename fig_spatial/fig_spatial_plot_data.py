import matplotlib.pyplot as plt
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, figure_style, save_figure_path
from fig_spatial.fig_spatial_load_data import load_data, load_dataframe, load_regions
import numpy as np
from matplotlib import cm, colors
import figrid as fg

fig_path = save_figure_path(figure='fig_spatial')


def plot_supp_figure():
    plot_fr_ff()
    plot_fr_3D()


def plot_fr_ff():
    df = load_dataframe()
    data = load_data(event='move', smoothing='sliding', norm=None)

    #df_filt = filter_recordings(df)
    df_filt = filter_recordings(df, min_regions=0, min_rec_lab=0, min_lab_region=0)
    all_frs_l = data['all_frs_l'][df_filt['include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['include'] == 1]
    all_ffs_l = data['all_ffs_l'][df_filt['include'] == 1]
    all_ffs_r = data['all_ffs_r'][df_filt['include'] == 1]
    df_filt = df_filt[df_filt['include'] == 1].reset_index()

    df_filt_reg = df_filt.groupby('region')
    
    # For setting y limits for the firing rate each brain region; May need to change:
    ylow = [3.35, 5.8, 5.6, 5.35, 6.3]
    yhigh = [7.4, 13.1, 13.2, 9, 11]
 
    # For setting y limits for the Fano Factor each brain region; May need to change:
    ylow2 = [1.24, 1.35, 1.39, 1.53, 1.46]
    yhigh2 = [1.45, 1.9, 1.81, 1.76, 1.7]
    
    for reg, regIdx in zip(BRAIN_REGIONS, np.arange(5)):
        fig, axs = plt.subplots(2, 2, figsize=(9,6))
        #fig.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.25)
        
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

        ax = axs[0][0]
        ax.fill_between(data['time_fr'], frs_l_mean - frs_l_std, frs_l_mean + frs_l_std, color='k', alpha=0.25)
        ax.errorbar(data['time_fr'], frs_l_mean, frs_l_std, color='k', capsize=2, linewidth=2.5, elinewidth=0.5)#, ecolor='gray')
        ax.set_ylabel('Firing Rate (Sp/s)', fontsize = 14)
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title('CW Movement Post Left Stim', fontsize = 14)
        #print(ax.get_ylim())
        ax.set_ylim(ylow[regIdx], yhigh[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='g', linestyle='dashed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax = axs[1][0]
        ax.fill_between(data['time_ff'], ffs_l_mean - ffs_l_std, ffs_l_mean + ffs_l_std, color='k', alpha=0.25)
        ax.errorbar(data['time_ff'], ffs_l_mean, ffs_l_std, color='k', capsize=2, linewidth=2.5, elinewidth=0.5)#, ecolor='gray')
        ax.set_ylabel('Fano Factor', fontsize = 14)
        ax.set_xlabel('Time from movement onset (s)', fontsize = 14)
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_ylim(ylow2[regIdx], yhigh2[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='g', linestyle='dashed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax = axs[0][1]
        ax.fill_between(data['time_fr'], frs_r_mean - frs_r_std, frs_r_mean + frs_r_std, color='k', alpha=0.25)
        ax.errorbar(data['time_fr'], frs_r_mean, frs_r_std, color='k', capsize=2, linewidth=2.5, elinewidth=0.5)#, ecolor='gray') 
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title('CCW Movement Post Right Stim', fontsize = 14)
        #print(ax.get_ylim())
        ax.set_ylim(ylow[regIdx], yhigh[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='g', linestyle='dashed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = axs[1][1]
        ax.fill_between(data['time_ff'], ffs_r_mean - ffs_r_std, ffs_r_mean + ffs_r_std, color='k', alpha=0.25, linewidth=0)
        ax.errorbar(data['time_ff'], ffs_r_mean, ffs_r_std, color='k', capsize=2, linewidth=2.5, elinewidth=0.5)#, ecolor='gray') 
        ax.set_xlabel('Time from movement onset (s)', fontsize = 14)
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_ylim(ylow2[regIdx], yhigh2[regIdx]) # May need to change
        ax.vlines(0, *ax.get_ylim(), color='g', linestyle='dashed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(fig_path.joinpath(f'fig_spatial_SuppFF_TimeCourse_{reg}.png'))


def plot_fr_3D():

    df = load_dataframe()
    #df_filt = filter_recordings(df)
    df_filt = filter_recordings(df, min_regions=0, min_rec_lab=0, min_lab_region=0)
    df_filt = df_filt[df_filt['include'] == 1].reset_index()

    reg_cent_of_mass = load_regions()

    df_filt_reg = df_filt.groupby('region')
    for reg in BRAIN_REGIONS:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        df_reg = df_filt_reg.get_group(reg)
        cofm_reg = reg_cent_of_mass[reg_cent_of_mass['region'] == reg]

        norm = colors.Normalize(vmin=0, vmax=3, clip=False)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))
        cluster_color = np.array([mapper.to_rgba(col) for col in df_reg['avg_ff_post_move']])
        s = np.ones_like(df_reg['x']) * 2
        s[df_reg['avg_ff_post_move'] < 1] = 10

        scat = ax.scatter((df_reg['x'].values - cofm_reg['x'].values) * 1e6, (df_reg['y'].values - cofm_reg['y'].values) * 1e6,
                          (df_reg['z'].values - cofm_reg['z'].values) * 1e6, c=cluster_color, marker='o', s=s)
        ax.scatter(0, 0, 0, c='r', marker='o', s=10)
        ax.scatter(0, 0, 0, c='r', marker='x', s=10)
        #cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label('Fano Factor')
        ax.set_xlabel(r'$\Delta$x')
        ax.set_ylabel(r'$\Delta$y')
        ax.set_zlabel(r'$\Delta$z', rotation=50)
        ax.view_init(15, 70)
        plt.draw()

        plt.savefig(fig_path.joinpath(f'fig_spatial_SuppFF_Spatial_{reg}.png'))


if __name__ == '__main__':
    plot_supp_figure()
