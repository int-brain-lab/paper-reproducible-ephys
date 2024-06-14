import numpy as np
import matplotlib.pyplot as plt

from iblatlas.atlas import AllenAtlas, Trajectory

from reproducible_ephys_functions import labs, figure_style, filter_recordings, save_figure_path
from fig_ephysfeatures.ephysfeatures_load_data import load_dataframe


def plot_figure_supp2(freeze=None):

    ba = AllenAtlas()

    lab_number_map, institution_map, lab_colors = labs()

    n_rec_per_lab = 0
    df_chns = load_dataframe(df_name='chns')

    df_filt = filter_recordings(min_rec_lab=n_rec_per_lab, min_neuron_region=0, freeze=freeze)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['institute', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('institute').size()
    df_filt['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

    rows = 4
    cols = int(np.ceil(len(df_filt['subject']) / rows))
    figure_style()
    fig = plt.figure(constrained_layout=False, figsize=(7, 5), dpi=300)
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=np.r_[np.ones(cols) * 0.9, 0.1])
    gs.update(wspace=0.05, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)

    ml_ranges = []
    axs = []
    cmin, cmax = np.quantile(ba.image, [0.1, 0.98])

    for iR, data in df_filt.iterrows():
        df = df_chns[df_chns['pid'] == data['pid']]

        xyz_chans = np.c_[df['x'], df['y'], df['z']] * 1e6
        xyz_samples = xyz_chans
        traj = Trajectory.fit(xyz_samples / 1e6)

        vector_perp = np.array([1, -1 * traj.vector[0] / traj.vector[2]])
        extent = 2000
        steps = np.ceil(extent * 2 / ba.res_um).astype(int)
        image = np.zeros((xyz_samples.shape[0], steps))

        ml_range = np.array([0, 0])
        dv_range = np.array([0, 0])
        for i, xyz in enumerate(xyz_samples):
            origin = np.array([xyz[0], xyz[2]])
            anchor = np.r_[[origin + extent * vector_perp], [origin - extent * vector_perp]]
            xargmin = np.argmin(anchor[:, 0])
            xargmax = np.argmax(anchor[:, 0])
            perp_ml = np.linspace(anchor[xargmin, 0], anchor[xargmax, 0], steps)
            perp_ap = np.ones_like(perp_ml) * xyz[1]
            perp_dv = np.linspace(anchor[xargmin, 1], anchor[xargmax, 1], steps)
            if i == 0:
                ml_range = [np.min(perp_ml), np.max(perp_ml)]
                dv_range = [np.min(perp_dv), np.max(perp_dv)]
            else:
                ml_range = [np.min(np.r_[ml_range[0], perp_ml]), np.max(np.r_[ml_range[1], perp_ml])]
                dv_range = [np.min(np.r_[dv_range[0], perp_dv]), np.max(np.r_[dv_range[1], perp_dv])]
            if dv_range[1] <= 0:
                idx = ba.bc.xyz2i(np.c_[perp_ml, perp_ap, perp_dv] / 1e6)
                idx[idx[:, 2] >= ba.image.shape[2], 2] = ba.image.shape[2] - 1
                image[i, :] = ba.image[idx[:, 1], idx[:, 0], idx[:, 2]]
        image = np.flipud(image)
        ml_ranges.append(ml_range)

        ax = fig.add_subplot(gs[int(iR / cols), np.mod(iR, cols)])

        ax.imshow(image, aspect='auto', extent=np.r_[[0, 4000], [0, 3840]], cmap='bone', alpha=1, vmin=cmin, vmax=cmax)
        scat = ax.scatter(df['lateral_um'] + 2000, df['axial_um'], c=df['lfp'], cmap='viridis', s=2, vmin=-190, vmax=-150)
        ax.set_title(data['recording'] + 1, color=lab_colors[data['institute']])
        ax.plot([200, 200], [100, 1100], color='red', lw=2)

        if i == 0:
            ax.tick_params(axis='y')
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_ylabel('Depth along probe (um)')
        else:
            ax.set_axis_off()

        ax.set(xticks=[])

        axs.append(ax)

    ax = fig.add_subplot(gs[0:rows, cols])
    cbar = fig.colorbar(scat, cax=ax, ticks=scat.get_clim())
    cbar.set_label('Power spectral density', rotation=270, labelpad=-8)
    cbar.ax.set_yticklabels([f'{-190} dB', f'{-150} dB'])

    save_path = save_figure_path(figure='fig_ephysfeatures')
    plt.savefig(save_path.joinpath('figure3_supp2.png'))
    plt.savefig(save_path.joinpath('figure3_supp2.pdf'))


if __name__ == '__main__':
    plot_figure_supp2(freeze='freeze_2024_03')
