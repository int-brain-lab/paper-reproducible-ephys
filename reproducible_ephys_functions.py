"""
General functions for reproducible ephys paper
"""

import seaborn as sns

def query():
    from oneibl.one import ONE
    one = ONE()
    trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                              x=-2243, y=-2000,  # repeated site coordinate
                              project='ibl_neuropixel_brainwide_01',
                              django='probe_insertion__session__qc__lt,50')  # All except CRITICAL
    return trajectories
    

def labs():
    lab_number_map = {'cortexlab': 'Lab 1', 'mainenlab': 'Lab 2', 'zadorlab': 'Lab 3',
                      'churchlandlab': 'Lab 4', 'angelakilab': 'Lab 5', 'wittenlab': 'Lab 6',
                      'hoferlab': 'Lab 7', 'mrsicflogellab': 'Lab 7', 'danlab': 'Lab 8',
                      'steinmetzlab': 'Lab 9'}
    institution_map = {'cortexlab': 'UCL', 'mainenlab': 'CCU', 'zadorlab': 'CSHL\n(Zador)',
                       'churchlandlab': 'CSHL\n(Churchland)', 'angelakilab': 'NYU',
                       'wittenlab': 'Princeton', 'hoferlab': 'SWC', 'mrsicflogellab': 'SWC',
                       'danlab': 'Berkeley',
                       'steinmetzlab': 'UW'}
    lab_colors = sns.color_palette("Dark2", 9)
    return lab_number_map, institution_map, lab_colors





