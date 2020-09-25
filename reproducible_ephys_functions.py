"""
General functions for reproducible ephys paper
"""

import seaborn as sns
from oneibl.one import ONE
one = ONE()


def query():
    trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                              x=-2243, y=-2000,  # repeated site coordinate
                              project='ibl_neuropixel_brainwide_01',
                              django='probe_insertion__session__qc__lt,50')  # All except CRITICAL
    return trajectories
    

def labs():
    institution_map = {'UCL': 'Lab 1', 'CCU': 'Lab 2', 'CSHL': 'Lab 3', 'NYU': 'Lab 4',
                       'Princeton': 'Lab 5', 'SWC': 'Lab 6', 'Berkeley': 'Lab 7'}
    color_palette = sns.color_palette("Dark2", 7)
    return institution_map, color_palette



