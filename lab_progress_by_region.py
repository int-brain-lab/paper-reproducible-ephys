from oneibl.one import ONE
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from ibllib.pipes.ephys_alignment import EphysAlignment
import numpy as np
from ibllib.ephys.neuropixel import SITES_COORDINATES
import pandas as pd
from reproducible_ephys_functions import STR_QUERY, BRAIN_REGIONS

# Initialise classes
one = ONE()
ba = AllenAtlas()
r = BrainRegions()


# Limit to resolved sessions
resolved = True
# Limit to sessions that pass behaviour
behavior = False

# Minimum number of channels needed to consider region targetting
min_ch = 5

str_query = STR_QUERY
if resolved:
    str_query = str_query + ',' + 'probe_insertion__json__extended_qc__alignment_resolved,True'
else:
    str_query = str_query + ',' + 'probe_insertion__json__extended_qc__tracing_exists,True'

if behavior:
    str_query = str_query + ',' + 'probe_insertion__session__extended_qc__behavior,1'


# All repeated site trajectories
trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=str_query)

# Loop through trajectories
dict_list = []
for traj in trajectories:
    ins = one.alyx.rest('insertions', 'list', id=traj['probe_insertion'])[0]
    xyz_picks = np.array(ins['json']['xyz_picks']) / 1e6

    if not resolved:
        # See if we have am alignment
        align_key = ins['json']['extended_qc'].get('alignment_stored', False)

        if not align_key:
            feature = None
            track = None
        else:
            align_traj = \
            one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                          probe_insertion=traj['probe_insertion'])[0]
            feature = np.array(align_traj['json'][align_key][0])
            track = np.array(align_traj['json'][align_key][1])

    else:
        align_traj = one.alyx.rest('trajectories', 'list',
                                   provenance='Ephys aligned histology track',
                                   probe_insertion=traj['probe_insertion'])[0]

        align_key = ins['json']['extended_qc']['alignment_stored']
        feature = np.array(align_traj['json'][align_key][0])
        track = np.array(align_traj['json'][align_key][1])

    ephysalign = EphysAlignment(xyz_picks, chn_depths=SITES_COORDINATES[:, 1], track_prev=track,
                                feature_prev=feature, brain_atlas=ba)

    ch_xyz = ephysalign.get_channel_locations(ephysalign.feature_init, ephysalign.track_init)
    ch_regions = ephysalign.get_brain_locations(ch_xyz)

    if ins['session_info']['lab'] == 'hoferlab':
        ins['session_info']['lab'] = 'mrsicflogellab'

    dict_frame = {'subjects': ins['session_info']['subject'],
                  'dates': ins['session_info']['start_time'][0:10],
                  'probes': ins['name'], 'lab': ins['session_info']['lab'], 'eid': ins['session'],
                  'probe_insertion': ins['id']}

    for reg in BRAIN_REGIONS:
        if reg in ['VISa', 'DG']:
            ch_regions['parent'][np.isnan(ch_regions['parent'])] = 997
            pa_regions = r.get(ch_regions['parent'])
            val = np.where(pa_regions['acronym'] == reg)[0].size
        else:
            val = np.where(ch_regions['acronym'] == reg)[0].size

        dict_frame[reg] = val

    dict_list.append(dict_frame)

# Make an overview data frame
df = pd.DataFrame(dict_list)
df = df.sort_values('lab')

# Now let's make a concise dataframe
lab_list = []
for lab in np.unique(df['lab'].values):
    df_lab = df.loc[df['lab'] == lab]
    lab_dict = {'lab': lab}
    for reg in BRAIN_REGIONS:
        val = np.sum(df_lab[reg] > min_ch)
        lab_dict[reg] = val

    lab_list.append(lab_dict)

df_lab_summary = pd.DataFrame(lab_list)
print(df_lab_summary)