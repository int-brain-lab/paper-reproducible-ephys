# %% !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:43:50 2020

@author: guido
"""

import pandas as pd
from reproducible_ephys_functions import data_path
from scratch_scripts.MF.features_2D import rms_data, get_brain_boundaries, plot_probe, psd_data
from one.api import ONE
import numpy as np
from os.path import join
from iblatlas.regions import BrainRegions
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
import matplotlib.pyplot as plt
from iblatlas import atlas
brain_atlas = atlas.AllenAtlas(25)
one = ONE()

# Plot settings
BOUNDARY = 'DG-TH'
PLOT = 'psd'
YLIM = [-1500, 1900]
FIG_SIZE = (10, 10)
SUBJECTS = ['DY_009', 'ZM_2241', 'CSHL059', 'CSH_ZAD_029', 'NYU-29',
            'ibl_witten_29', 'SWC_054', 'KS023']
CMAPS = ['gist_stern', 'gist_rainbow', 'cubehelix', 'turbo', 'gnuplot',
         'gist_earth', 'brg', 'rainbow']

# Load in recordings
data = pd.read_csv(join(data_path(), 'metrics_region.csv'))

# Include recordings
data = data[data['subject'].isin(SUBJECTS)]
data = data.drop_duplicates(subset='subject').reset_index()


# %% Plotting

f, axs = plt.subplots(1, len(SUBJECTS), figsize=FIG_SIZE)
for i, ax in enumerate(axs):

    # Load in recording data
    eid = one.search(subject=data.loc[i, 'subject'], date=data.loc[i, 'date'])[0]
    probe_label = data.loc[i, 'probe']
    if i == 0:
        chn_inds = one.load_dataset(eid, dataset=['channels.rawInd.npy'],
                                    collection=f'alf/{probe_label}')
    ephys_path = one.eid2path(eid).joinpath('raw_ephys_data', probe_label)

    # Get depth alignment
    insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
    align_key = insertion[0].get('json').get('extended_qc').get('alignment_stored', None)
    trajectory = one.alyx.rest('trajectories', 'list',
                               provenance='Ephys aligned histology track',
                               probe_insertion=insertion[0]['id'])
    alignments = trajectory[0]['json']
    track = np.array(alignments[align_key][1])
    feature = np.array(alignments[align_key][0])
    depths = SITES_COORDINATES[:, 1]
    xyz_picks = np.array(insertion[0].get('json').get('xyz_picks', 0)) / 1e6
    ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track,
                                feature_prev=feature,
                                brain_atlas=brain_atlas)
    xyz_channels = ephysalign.get_channel_locations(feature, track)
    z = xyz_channels[:, 2] * 1e6
    r = BrainRegions()
    brain_regions = ephysalign.get_brain_locations(xyz_channels)
    boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)
    z_subtract = boundaries[np.where(np.array(regions) == BOUNDARY)[0][0] + 1]
    z = z - z_subtract

    # Call plotting functions
    if PLOT == 'rms_ap':
        plot_data = rms_data(ephys_path, one, eid, chn_inds, 'AP')
    elif PLOT == 'psd':
        plot_data = psd_data(ephys_path, one, eid, chn_inds)
    im = plot_probe(plot_data, z, ax, cmap=CMAPS[i])
    ax.set(ylim=YLIM)
    ax.axis('off')





