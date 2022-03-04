from reproducible_ephys_functions import get_insertions, combine_regions, BRAIN_REGIONS, labs
from one.api import ONE
from brainbox.processing import compute_cluster_average
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
import numpy as np
from ibllib.pipes.ephys_alignment import EphysAlignment
import pandas as pd
from reproducible_ephys_paths import DATA_PATH
from pathlib import Path
from one.alf.exceptions import ALFObjectNotFound

one = ONE()
ba = AllenAtlas()
lab_number_map, institution_map, lab_colors = labs()



insertions = get_insertions(level=2)
download_waveforms = False
LFP_BAND_HIGH = [20, 80]
LFP_BAND_LOW = [2, 15]

all_df_clust = []
all_df_chns = []
for iIns, ins in insertions:

    print(f'processing {iIns + 1}/{len(insertions)}')
    data_clust = {}
    data_chns = {}
    data_ins = {}

    pid = ins['probe_insertion']
    eid = ins['session']['id']
    probe = ins['probe_name']
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
    channels['rawInd'] = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=sl.collection)
    clusters = sl.merge_clusters(spikes, clusters, channels)

    channels['rep_site_acronym'] = combine_regions(channels['acronym'])
    channels['rep_site_acronym_alt'] = np.copy(channels['rep_site_acronym'])
    channels['rep_site_acronym_alt'][channels['rep_site_acronym_alt'] == 'PPC'] = 'VISa'
    channels['rep_site_id'] = ba.regions.acronym2id(channels['rep_site_acronym_alt'])
    clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])

    # Data for cluster dataframe
    cluster_idx = np.where(clusters['label'] == 1)
    data_clust['cluster_ids'] = clusters['cluster_id'][cluster_idx]
    data_clust['region'] = clusters['rep_site_acronym'][cluster_idx]
    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], data_clust['cluster_ids'])

    clu, data_clust['depths'], n_cluster = compute_cluster_average(spikes['clusters'][spike_idx], spikes['depths'][spike_idx])
    assert np.array_equal(clu, data_clust['cluster_ids'])
    clu, data_clust['amps'], _ = compute_cluster_average(spikes['clusters'][spike_idx], spikes['amps'][spike_idx])
    assert np.array_equal(clu, data_clust['cluster_ids'])
    data_clust['fr'] = n_cluster / np.max(spikes.times) - np.min(spikes.times)

    insertion = one.alyx.rest('insertions', 'list', id=pid)[0]
    traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track', probe_insertion=pid)[0]
    feature, track = [*traj['json'][insertion['json']['extended_qc']['alignment_stored']]][:2]
    xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6
    ephysalign = EphysAlignment(xyz_picks, channels['axial_um'], brain_atlas=ba, feature_prev=feature, track_prev=track,
                                speedy=True)
    data_clust['depths_aligned'] = ephysalign.get_channel_locations(feature, track, data_clust['depths'] / 1e6)[:, 2] * 1e6

    df_clust = pd.DataFrame.from_dict(data_clust)
    df_clust['eid'] = eid
    df_clust['pid'] = pid
    df_clust['subject'] = ins['session']['subject']
    df_clust['probe'] = ins['probe_name']
    df_clust['date'] = ins['session']['start_time'][:10]
    df_clust['lab'] = ins['session']['lab']
    df_clust['institute'] = df_clust['lab'].map(institution_map)
    df_clust['lab_number'] = df_clust['lab'].map(lab_number_map)

    # Data for channel dataframe
    lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
    freqs = ((lfp['freqs'] > LFP_BAND_HIGH[0])
             & (lfp['freqs'] < LFP_BAND_HIGH[1]))
    power = lfp['power'][:, channels['rawInd']]
    lfp_power = np.nanmean(10 * np.log(power[freqs]), axis=0)

    data_chns['x'] = channels['x']
    data_chns['y'] = channels['y']
    data_chns['z'] = channels['z']
    data_chns['lfp'] = lfp_power
    data_chns['region_id'] = channels['atlas_id']
    data_chns['region_id_rep'] = channels['rep_site_id']
    data_chns['region'] = channels['rep_site_acronym']

    df_chns = pd.DataFrame.from_dict(data_chns)
    df_chns['eid'] = eid
    df_chns['pid'] = pid
    df_chns['subject'] = ins['session']['subject']
    df_chns['probe'] = ins['probe_name']
    df_chns['date'] = ins['session']['start_time'][:10]
    df_chns['lab'] = ins['session']['lab']
    df_chns['institute'] = df_chns['lab'].map(institution_map)
    df_chns['lab_number'] = df_chns['lab'].map(lab_number_map)

    # Now for the insertions (minus LFP at the moment)

    all_df_clust.append(df_clust)
    all_df_chns.append(df_chns)

concat_df_clust = pd.concat(all_df_clust, ignore_index=True)
concat_df_chns = pd.concat(all_df_chns, ignore_index=True)
save_path = Path(DATA_PATH).joinpath('figure3')
save_path.mkdir(exist_ok=True, parents=True)
concat_df_clust.to_csv(save_path.joinpath('figure3_dataframe_clust.csv'))
concat_df_chns.to_csv(save_path.joinpath('figure3_dataframe_chns.csv'))










    # lfp_power = lfp_power[:, chn_inds]
#
    # # Define a frequency range of interest
    # freq_idx = np.where((lfp_freq >= freq_range[0]) &
    #                     (lfp_freq < freq_range[1]))[0]
#
    # # Limit data to freq range of interest and also convert to dB
    # lfp_spectrum_data = 10 * np.log(lfp_power[freq_idx, :])
    # lfp_spectrum_data[np.isinf(lfp_spectrum_data)] = np.nan
    # lfp_mean = np.mean(lfp_spectrum_data, axis=0)
#
#
#
#
#
#
#
    # channels['rep_site_acronym'] = combine_regions(channels['acronym'])
    # channels['rep_site_acronym_alt'] = np.copy(channels['rep_site_acronym'])
    # channels['rep_site_acronym_alt'][channels['rep_site_acronym_alt'] == 'PPC'] = 'VISa'
    # channels['rep_site_id'] = ba.regions.acronym2id(channels['rep_site_acronym_alt'])
    # channels['beryl_acronym'] = ba.regions.acronym2acronym(channels['acronym'], mapping='Beryl')
    # channels['beryl_id'] = ba.regions.id2id(channels['atlas_id'], mapping='Beryl')
#
    # lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
#
    # boundaries, colours, regions = get_brain_boundaries(channels['atlas_id'], channels['z'] * 1e6, ba.regions)

