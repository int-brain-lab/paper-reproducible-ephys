'''
Compute / load metrics table to then filter according to criteria
RE TF
'''
##
from matplotlib.sankey import Sankey
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from one.api import ONE
from pathlib import Path
from reproducible_ephys_functions import load_metrics, compute_metrics, filter_recordings, get_insertions
##
one = ONE()
freeze = 'freeze_2024_03'

save_path_sankey = Path('/Users/gaelle/Desktop/sankey_manual_05-06-2024.pdf')

# Potential PIDs
df_ins_good = filter_recordings(min_regions=-1, freeze=freeze)
df_ins_good = df_ins_good.drop_duplicates('pid')
# Included in analysis
include = df_ins_good[df_ins_good['include'] == True].pid.unique()

##
# Get PIDs of critical insertions

# Insertion CRITICAL
django_queries = []
django_queries.append('probe_insertion__session__projects__name,ibl_neuropixel_brainwide_01')
django_queries.append('probe_insertion__json__qc,CRITICAL')
django_query = ','.join(django_queries)

trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=django_query)
pids_ic = [traj['probe_insertion'] for traj in trajectories]

# Session CRITICAL
django_queries = []
django_queries.append('probe_insertion__session__projects__name,ibl_neuropixel_brainwide_01')
django_queries.append('probe_insertion__session__qc__gte,50')
django_query = ','.join(django_queries)

trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=django_query)
pids_sc = [traj['probe_insertion'] for traj in trajectories]

# Unique set
pids_crt = set(pids_ic).union(set(pids_sc))

# Remove critical PIDs for which we know we don't need to recompute as fail for other reasons:
# Reasons are : Hardware failure of tracing missing --> left with Ephys issues
remove_crt_histology = [
    '3443eceb-50b3-450e-b7c1-fc465a3bc84f',
    '553f8d56-b6e7-46bd-a146-ac43b8ec6de7',
    '6bfa4a99-bdfa-4e44-aa01-9c7eac7e253d',
    '82a42cdf-3140-427b-8ad0-0d504716c871',
    'dc6eea9b-a8fb-4151-b298-718f321e6968',
    '81347b01-b2a2-4985-b1b1-436fb83049d1',
    '978b4073-e3b0-440a-a336-cd26e3bae8ea',
    'c5130b6f-f584-4514-8be4-22bcde848a60'
    ]

remove_crt_hardware = [
    '143cff36-30b4-4b25-9d26-5a2dbe6f6fc2',
    '79bcfe47-33ed-4432-a598-66006b4cde56',
    '80624507-4be6-4689-92df-0e2c26c3faf3',
    'e7abb87f-4324-4c89-8a46-97ed4b40577e',
    'f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c',
    'fc626c12-bd1e-45c3-9434-4a7a8c81d7c0',
    'd8ff1218-75e1-4962-b920-98c40b9dea1a',
    '41a3b948-13f4-4be7-90b9-150705d39005',
    '99b8eb91-393d-4ed1-913e-ce5ee2e31bc3',
    '9e44ddb5-7c7c-48f1-954a-6cec2ad26088'
]

pids_crt_ephysonly = pids_crt - set(remove_crt_histology) - set(remove_crt_hardware)
# These need to be checked manually for subclasses of Ephys errors
# Hence the Sankey plot has manual values in

##
# Generate numbers for Sankey plot
n_all_ins = len(pids_crt) + df_ins_good.shape[0]
n_hw_crt = len(remove_crt_hardware)
n_hist_crt = len(remove_crt_histology)
n_ephys_crt = len(pids_crt_ephysonly)

# ----
# Low yield
indx_low_yield = np.where(df_ins_good['low_yield'])[0]  # TODO do not know how to return indx from drop
# High noise is the mixture of both LFP and AP  noise
indx_high_noise = np.where((df_ins_good['high_noise'] == True) | (df_ins_good['high_lfp'] == True))[0]
# Behavior (N trial)
indx_behav = np.where(df_ins_good['low_trials'] == True)[0]

# Check that there are no overlap to count numbers
n_low_yield = len(indx_low_yield)
# We remove from the noise those that already were down due to low yield
set_high_noise = set(indx_high_noise) - set(indx_low_yield)
n_high_noise = len(set_high_noise)
# We remove from Behavior those failing previously
set_behavior = set(indx_behav) - (set(indx_high_noise).union(set(indx_low_yield)))
n_behav = len(set_behavior)

##
# Sankey plot
sns.set_theme()

# Manually assign the numbers below
ephys_crt_drift = 1
ephys_crt_noisych = 10
ephys_crt_artefact = 2
ephys_crt_epilepsy = 1
assert n_ephys_crt == ephys_crt_drift+ephys_crt_noisych+ephys_crt_artefact+ephys_crt_epilepsy

data_analysis = include.shape[0]

num_trajectories = np.array([
    n_all_ins,
    -n_behav,
    -n_hist_crt, -n_hw_crt,
    -ephys_crt_drift, -ephys_crt_noisych, -ephys_crt_artefact, -ephys_crt_epilepsy,
    -n_low_yield, -n_high_noise,
    -data_analysis
    ])
portion_remove = num_trajectories/n_all_ins  # remove portion from trunk length of sankey plot otherwise not aligned
assert sum(num_trajectories) == 0

orientations = np.array([
    0,  # all in straight
    -1, -1,  # critical hardware and histology down arrow
    -1,  # -1, -1,   # behav and missed target down arrow
    1, 1, 1, 1,  # critical ephys up arrow (Table 1)
    1, 1,  # low yield and high noise up arrow (Table 1)
    0  # remaining straight
    ])

portion_remove[:] = 0
default_val = 0.2
concat1 = np.array([0.08, default_val])
concat2 = default_val - portion_remove[1:-2]
concat3 = np.array([0.4])
pathlengths = np.concatenate((concat1, concat2, concat3), axis=0)

labels = ['All insertions',
          'Poor behavior',  # 'Missed target', 'Poor behavior',
          'Missing histology', 'Hardware failure',
          'Poor ephys (Drift)', 'Poor ephys (Noisy channels)', 'Poor ephys (Artefact)', 'Poor ephys (Epileptiform)',
          'Low neural yield', 'High noise',
          'Data analysis']

fig, ax = plt.subplots()
sankey = Sankey(ax=ax, scale=0.005, offset=0.05, head_angle=90, shoulder=0.025, gap=0.2, radius=0.05)
sankey.add(flows=num_trajectories,
           labels=labels,
           trunklength=0.9,
           orientations=orientations,
           pathlengths=pathlengths,
           facecolor=sns.color_palette('Pastel1')[1])
diagrams = sankey.finish()

# text font and positioning
for text in diagrams[0].texts:
    text.set_fontsize('7')

text = diagrams[0].texts[0]
xy = text.get_position()
text.set_position((xy[0] - 0.3, xy[1]))
text.set_weight('bold')

text = diagrams[0].texts[-1]
xy = text.get_position()
text.set_position((xy[0] + 0.2, xy[1]))
text.set_weight('bold')

ax.axis('off')

plt.savefig(save_path_sankey, bbox_inches='tight')
