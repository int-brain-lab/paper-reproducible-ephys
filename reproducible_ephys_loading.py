import brainbox.io.one as bbone
from ibllib.atlas import AllenAtlas
import os
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
from iblutil.numerical import ismember
from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS
import logging

logger = logging.getLogger('ibllib')


from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader


FIGURE_1 =


FIGURE_2 = {'channels': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv', 'rawInd'],
            'electrodeSites': ['localCoordinates', 'brainLocationIds_ccf_2017', 'mlapdv'],}

FIGURE_3 = {'channels': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv', 'rawInd'],
            'clusters': ['channels', 'metrics'],
            'electrodeSites': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv'],
            'ephysSpectralDensityLF': ['freqs', 'power'],
            'ephysTimeRms': ['rms'],
            'spikes': ['amps', 'clusters', 'depths', 'times']}

FIGURE_4 = {'channels': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv', 'rawInd'],
            'clusters': ['channels', 'metrics'],
            'spikes': ['amps', 'clusters', 'depths', 'times'],
            'trials': ['choice', 'contrastLeft', 'contrastRight', 'feedbackType', 'firstMovement_times', 'stimOn_times']}

FIGURE_5 = {'channels': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv', 'rawInd'],
            'clusters': ['amps', 'channels', 'metrics', 'peak2trough'],
            'spikes': ['amps', 'clusters', 'depths', 'times'],
            'trials': ['choice', 'contrastLeft', 'contrastRight', 'feedbackType', 'feedback_times',
                       'firstMovement_times', 'stimOn_times']}









def get_insertions(level=1, force_reload=False):
    pass


def download_spike_sorting_data(pid, one=None, brain_atlas=None):
    one = one or ONE()
    ba = brain_atlas or AllenAtlas()
    sl = SpikeSortingLoader(pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels, cache_dir=sl.spike_sorting_path)

    return spikes, clusters, channels


# figure 4, 5 spikes + trials

def download_figure_data(pids, figure=1, one=None, brain_atlas=None):
    one = one or ONE()
    ba = brain_atlas or AllenAtlas()

    for pid in pids:
        # load spike sorting
        _, _ _, = download_spike_sorting_data()
        sl = SpikeSortingLoader(pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels, cache_dir=sl.spike_sorting_path)










def compute_metrics_df(trajectories, one=None, ba=None):
    one = one or ONE()
    ba = ba or AllenAtlas()

    for i, traj in enumerate(trajectories):
        metrics = pd.DataFrame()
        eid = traj['session']['id']
        lab = traj['session']['lab']
        nickname = traj['session']['subject']
        date = traj[i]['session']['start_time'][:10]
        pid = traj['probe_insertion']
        probe = traj['probe_name']

        collection = f'alf/{probe}/pykilosort'

        try:
            ap = one.load_object(eid, 'ephysChannels', collection=f'raw_ephys_data/{probe}', attribute=['apRMS'])
        except ALFObjectNotFound:
            logger.warning(f'ephysChannels object not found for pid: {pid}')
            ap = {}

        try:
            lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
        except ALFObjectNotFound:
            logger.warning(f'ephysSpectralDensityLF object not found for pid: {pid}')
            lfp = {}

        try:
            clusters = one.load_object(eid, 'clusters', collection=collection, attribute=['metrics', 'channels'])
            if 'metrics' not in clusters.keys():
                spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, probe=probe,  )
            else:
                channels = bbone.load_channel_locations(eid, probe=probe, one=one, aligned=True, brain_atlas=ba)[probe]
                channels['rawInd'] = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=collection)
                channels['rep_site_acronym'] = combine_regions(channels['acronym'])
                clusters['rep_site_acronym'] = channels['rep_site_acronym'][clusters['channels']]
        except Exception as err:
            print(err)


        for region in BRAIN_REGIONS:
            region_clusters = np.where(np.bitwise_and(clusters['rep_site_acronym'] == region,
                                                      clusters['metrics']['label'] == 1))[0]
            region_chan = channels['rawInd'][np.where(channels['rep_site_acronym'] == region)[0]]

            if 'power' in lfp.keys():
                freqs = ((lfp['freqs'] > LFP_BAND_HIGH[0])
                         & (lfp['freqs'] < LFP_BAND_HIGH[1]))
                chan_power = lfp['power'][:, region_chan]
                lfp_high_region = np.mean(10 * np.log(chan_power[freqs]))  # convert to dB
            else:
                lfp_high_region = np.nan

            if 'apRMS' in ap.keys() and region_chan.shape[0] > 0:
                ap_rms = np.percentile(ap['apRMS'][1, region_chan], 90) * 1e6
            else:
                ap_rms = 0

            metrics = metrics.append(pd.DataFrame(
                index=[metrics.shape[0] + 1], data={'pid': pid, 'eid': eid, 'probe': probe,
                                                    'lab': lab, 'subject': nickname,
                                                    'region': region, 'date': date,
                                                    'n_channels': region_chan.shape[0],  # THIS ONE
                                                    'neuron_yield': region_clusters.shape[0],  # THIS ONE
                                                    'lfp_power_high': lfp_high_region,  # THIS ONE
                                                    'lfp_band_high': [LFP_BAND_HIGH],
                                                    'rms_ap_p90': ap_rms}))

        metrics.attrs['time']
        metrics.attrs['created_by'] = one.user_name



