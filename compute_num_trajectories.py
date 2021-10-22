# -*- coding: utf-8 -*-
"""

@author: Noam Roth

Get numbers of sessions for reproducible ephys query and determine where they are failing
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
from one.api import ONE


#basic BWM query
STR_QUERY = 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01' 
BRAIN_REGIONS = ['VIS', 'CA1', 'DG', 'LP', 'PO']
one = ONE()

#Trajectories planned for rep. site
str_query = STR_QUERY
trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=str_query)
print('Number of total trajectories planned for repeated site: ', len(trajectories))


# With QC < 50
previous_num = len(trajectories) #save this number to compute the next decrease
str_query = str_query + ',' + 'probe_insertion__session__qc__lt,50' 
trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=str_query)
print('With QC < 50: ', len(trajectories), '(', len(trajectories) - previous_num, ')')

# With QC not critical
previous_num = len(trajectories) #save this number to compute the next decrease
str_query = str_query + ',' + '~probe_insertion__json__qc,CRITICAL' 
trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=str_query)
print('With QC not CRITICAL: ', len(trajectories), '(', len(trajectories) - previous_num, ')')

# With behavior criterion trials >=400 (relaxed criterion for this TF)
previous_num = len(trajectories) #save this number to compute the next decrease
str_query = str_query + ',' + 'probe_insertion__session__n_trials__gte,400'
trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=str_query)
print('With behavior criterion trials >=400:  ', len(trajectories), '(', len(trajectories) - previous_num, ')')


# With resolved alignment
previous_num = len(trajectories) #save this number to compute the next decrease
str_query = str_query + ',' + 'probe_insertion__json__extended_qc__alignment_resolved,True'
trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000, theta=15,
                             django=str_query)
print('With resolved alignment: ', len(trajectories), '(', len(trajectories) - previous_num, ')')


# With min regions >=3 
previous_num = len(trajectories) #save this number to compute the next decrease
min_regions = 3
region_traj = []
query_regions = BRAIN_REGIONS.copy()
for i, region in enumerate(query_regions):
    region_query = one.alyx.rest(
                'trajectories', 'list', provenance='Ephys aligned histology track',
                django=(str_query + ',channels__brain_region__acronym__icontains,%s' % region))
    region_traj = np.append(region_traj, [i['probe_insertion'] for i in region_query])
num_regions = np.empty(len(trajectories))
for i, trajectory in enumerate(trajectories):
    num_regions[i] = sum(trajectory['probe_insertion'] in s for s in region_traj)
trajectories = [trajectories[i] for i in np.where(num_regions >= min_regions)[0]]

print('With min regions >=3: ', len(trajectories), '(', len(trajectories) - previous_num, ')')

