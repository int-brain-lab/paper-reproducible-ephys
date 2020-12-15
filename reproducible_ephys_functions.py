"""
General functions for reproducible ephys paper
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib


def labs():
    lab_number_map = {'cortexlab': 'Lab 1', 'mainenlab': 'Lab 2', 'zadorlab': 'Lab 3',
                      'churchlandlab': 'Lab 4', 'angelakilab': 'Lab 5', 'wittenlab': 'Lab 6',
                      'hoferlab': 'Lab 7', 'mrsicflogellab': 'Lab 7', 'danlab': 'Lab 8',
                      'steinmetzlab': 'Lab 9'}
    institution_map = {'cortexlab': 'UCL', 'mainenlab': 'CCU', 'zadorlab': 'CSHL (Z)',
                       'churchlandlab': 'CSHL (C)', 'angelakilab': 'NYU',
                       'wittenlab': 'Princeton', 'hoferlab': 'SWC', 'mrsicflogellab': 'SWC',
                       'danlab': 'Berkeley',
                       'steinmetzlab': 'UW'}
    colors = sns.color_palette("Dark2", 10)[1:]
    institutions = ['UCL', 'CCU', 'CSHL (Z)', 'CSHL (C)', 'NYU', 'Princeton', 'SWC', 'Berkeley',
                    'UW']
    institution_colors = {}
    for i, inst in enumerate(institutions):
        institution_colors[inst] = colors[i]
    return lab_number_map, institution_map, institution_colors


def query(resolved=True, behavior=False, as_dataframe=False):
    from oneibl.one import ONE
    one = ONE()
    str_query = 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,' \
                 'probe_insertion__session__qc__lt,50'
    if resolved:
        str_query = str_query + ',' + 'probe_insertion__json__extended_qc__alignment_resolved,True'
    if behavior:
        str_query = str_query + ',' + 'probe_insertion__session__extended_qc__behavior,1'

    trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                 x=-2243, y=-2000,
                                 django=str_query)
    if as_dataframe:
        trajectories = pd.DataFrame(data={
                            'subjects': [i['session']['subject'] for i in trajectories],
                            'dates': [i['session']['start_time'][:10] for i in trajectories],
                            'probes': [i['probe_name'] for i in trajectories],
                            'lab': [i['session']['lab'] for i in trajectories],
                            'eid': [i['session']['id'] for i in trajectories]})
        institution_map = labs()[1]
        trajectories['institution'] = trajectories.lab.map(institution_map)
    return trajectories


def data_path():
    # Retrieve absolute path of paper-behavior dir
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(repo_dir, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return data_dir


def figure_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Helvetica",
            rc={"font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "lines.linewidth": 1,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42





