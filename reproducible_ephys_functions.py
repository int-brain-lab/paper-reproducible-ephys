"""
General functions for reproducible ephys paper
"""

import os
import seaborn as sns
import matplotlib


def query(resolved=True):
    from oneibl.one import ONE
    one = ONE()
    if resolved:
        trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                     x=-2243, y=-2000,  # repeated site coordinate
                                     project='ibl_neuropixel_brainwide_01',
                                     django='probe_insertion__session__qc__lt,50',  # excl CRITICAL
                                     histology=True)
        resolved_insertions = one.alyx.rest('insertions', 'list',
                                            provenance='Ephys aligned histology track',
                                            django='json__extended_qc__alignment_resolved,True')
        resolved_eids = [i['session'] for i in resolved_insertions]
        trajectories = [i for i in trajectories if str(i['session']['id']) in resolved_eids]
    else:
        trajectories = one.alyx.rest('insertions', 'list', provenance='Planned',
                                     x=-2243, y=-2000,  # repeated site coordinate
                                     project='ibl_neuropixel_brainwide_01',
                                     django='probe_insertion__session__qc__lt,50,',
                                     histology=True)
    return trajectories


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





