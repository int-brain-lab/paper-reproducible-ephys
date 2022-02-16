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


FIGURE_1 = {}


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

FIGURE_6 = {'channels': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv', 'rawInd'],
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



