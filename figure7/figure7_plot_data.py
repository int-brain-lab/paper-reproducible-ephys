import matplotlib.pyplot as plt
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS
from figure7.figure7_load_data import load_data, load_dataframe
import numpy as np
from matplotlib import cm, colors

df = load_dataframe()
data = load_data(event='move', smoothing='sliding', norm=None)

df_filt = filter_recordings(df)
all_frs_l = data['all_frs_l'][df_filt['include'] == 1]
all_frs_r = data['all_frs_r'][df_filt['include'] == 1]
all_ffs_l = data['all_ffs_l'][df_filt['include'] == 1]
all_ffs_r = data['all_ffs_r'][df_filt['include'] == 1]
df_filt = df_filt[df_filt['include'] == 1].reset_index()


df_filt_reg = df_filt.groupby('region')

for reg in BRAIN_REGIONS:
    fig, ax = plt.subplots(2, 2)
    df_reg = df_filt_reg.get_group(reg)
    reg_idx = df_filt_reg.groups[reg]
    ffs_r_reg = all_ffs_r[reg_idx, :]
    frs_r_reg = all_frs_r[reg_idx, :]
    ffs_l_reg = all_ffs_l[reg_idx, :]
    frs_l_reg = all_frs_l[reg_idx, :]
    ax[0][0].plot(data['time_fr'], np.nanmean(frs_r_reg, axis=0))
    ax[1][0].plot(data['time_ff'], np.nanmean(ffs_r_reg, axis=0))
    ax[0][1].plot(data['time_fr'], np.nanmean(frs_l_reg, axis=0))
    ax[1][1].plot(data['time_ff'], np.nanmean(ffs_l_reg, axis=0))
    fig.suptitle(reg)


for reg in BRAIN_REGIONS:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    df_reg = df_filt_reg.get_group(reg)
    norm = colors.Normalize(vmin=np.nanmin(df_reg['avg_ff_post_move']), vmax=np.nanmax(df_reg['avg_ff_post_move']), clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))
    cluster_color = np.array([mapper.to_rgba(col) for col in df_reg['avg_ff_post_move']])
    s = np.ones_like(df_reg['x']) * 2
    s[df_reg['avg_ff_post_move'] < 1] = 6
    scat = ax.scatter(df_reg['x'], df_reg['y'], df_reg['z'], c=cluster_color, marker='o', s=s)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
