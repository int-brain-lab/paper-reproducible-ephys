#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:26:39 2020

@author: guido
"""

import brainbox.io.one as bbone
from reproducible_ephys_functions import query
from reproducible_ephys_paths import FIG_PATH

# Query repeated site trajectories
rep_sites = query()

for i in range(len(rep_sites)):
    print('Processing session %d of %d' % (i+1, len(rep_sites)))
    
    # Load in data
    eid = rep_sites[i]['session']['id']
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
    except:
        continue
       
    # Get coordinates of micro-manipulator and histology
    hist = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                     probe_insertion=rep_site[i]['probe_insertion'])
    if len(hist) == 0:
        continue
    hit_regions.loc[eid, 'x_hist'] = hist[0]['x']
    hit_regions.loc[eid, 'y_hist'] = hist[0]['y']
    manipulator = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator',
                                probe_insertion=rep_site[i]['probe_insertion'])
    if len(manipulator) > 0:
        hit_regions.loc[eid, 'x_target'] = manipulator[0]['x']
        hit_regions.loc[eid, 'y_target'] = manipulator[0]['y']