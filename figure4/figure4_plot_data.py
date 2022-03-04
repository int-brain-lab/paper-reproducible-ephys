import matplotlib.pyplot as plt
import numpy as np

from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs
from figure4.figure4_load_data import load_data, load_dataframe

lab_number_map, institution_map, lab_colors = labs()

df = load_dataframe()
data = load_data(event='move', norm='subtract', smoothing='kernel')

df_filt = filter_recordings(df)
df_filt = df_filt[df_filt['include'] == 1]


# Example to get similar plot to figure 4b
fig, ax = plt.subplots()
subject = 'SWC_054'
region = 'CA1'
idx = df_filt.loc[(df['region'] == region) & (df['subject'] == subject)].index
time = data['time']
for fr, fr_std in zip(data['all_frs_l'][idx], data['all_frs_l_std'][idx]):
    ax.plot(time, fr, 'k')
    # ax.fill_between(time, fr-fr_std, fr+fr_std, 'k', alpha=0.5)

fr_mean = np.mean(data['all_frs_l'][idx], axis=0)
fr_std = np.std(data['all_frs_l'][idx], axis=0)
ax.plot(time, fr_mean, 'g')
# ax.fill_between(time, fr_mean - fr_std, fr_mean + fr_std, 'g', alpha=0.5)

# Example to get similar plot to figure 4c
fig, ax = plt.subplots(1, len(BRAIN_REGIONS))
df_filt_reg = df_filt.groupby('region')
for iR, reg in enumerate(BRAIN_REGIONS):
    df_reg = df_filt_reg.get_group(reg)
    df_reg_subj = df_reg.groupby('subject')
    for subj in df_reg_subj.groups.keys():
        df_subj = df_reg_subj.get_group(subj)
        subj_idx = df_reg_subj.groups[subj]
        frs_subj = data['all_frs_l'][subj_idx, :]
        ax[iR].plot(np.mean(frs_subj, axis=0), c=lab_colors[df_subj.iloc[0]['institute']])

