"""
General functions for reproducible ephys paper
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
from one.api import ONE

STR_QUERY = 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,' \
            'probe_insertion__session__qc__lt,50,' \
            '~probe_insertion__json__qc,CRITICAL,' \
            'probe_insertion__session__n_trials__gte,400'

BRAIN_REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']

def labs():
    lab_number_map = {'cortexlab': 'Lab 1', 'mainenlab': 'Lab 2', 'zadorlab': 'Lab 3',
                      'churchlandlab': 'Lab 4', 'angelakilab': 'Lab 5', 'wittenlab': 'Lab 6',
                      'hoferlab': 'Lab 7', 'mrsicflogellab': 'Lab 7', 'danlab': 'Lab 8',
                      'steinmetzlab': 'Lab 9', 'churchlandlab_ucla': 'Lab 10'}
    institution_map = {'cortexlab': 'UCL', 'mainenlab': 'CCU', 'zadorlab': 'CSHL (Z)',
                       'churchlandlab': 'CSHL (C)', 'angelakilab': 'NYU',
                       'wittenlab': 'Princeton', 'hoferlab': 'SWC', 'mrsicflogellab': 'SWC',
                       'danlab': 'Berkeley', 'steinmetzlab': 'UW', 'churchlandlab_ucla': 'UCLA'}
    colors = sns.color_palette("Dark2", 11)[1:]
    institutions = ['UCL', 'CCU', 'CSHL (Z)', 'CSHL (C)', 'NYU', 'Princeton', 'SWC', 'Berkeley',
                    'UW', 'UCLA']
    institution_colors = {}
    for i, inst in enumerate(institutions):
        institution_colors[inst] = colors[i]
    return lab_number_map, institution_map, institution_colors


def query(resolved=True, behavior=False, min_regions=2, as_dataframe=False, str_query=STR_QUERY,
          one=None):
    if one is None:
        one = ONE()

    # Query repeated site recordings
    if resolved:
        str_query = str_query + ',' + 'probe_insertion__json__extended_qc__alignment_resolved,True'
    if behavior:
        str_query = str_query + ',' + 'probe_insertion__session__extended_qc__behavior,1'

    trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                 x=-2243, y=-2000, theta=15,
                                 django=str_query)

    # Query how many of the target regions were hit per recording
    region_traj = []
    query_regions = BRAIN_REGIONS.copy()
    query_regions[query_regions.index('PPC')] = 'VIS'
    for i, region in enumerate(query_regions):
        region_query = one.alyx.rest(
                    'trajectories', 'list', provenance='Ephys aligned histology track',
                    django=(str_query + ',channels__brain_region__acronym__icontains,%s' % region))
        region_traj = np.append(region_traj, [i['probe_insertion'] for i in region_query])
    num_regions = np.empty(len(trajectories))
    for i, trajectory in enumerate(trajectories):
        num_regions[i] = sum(trajectory['probe_insertion'] in s for s in region_traj)
    trajectories = [trajectories[i] for i in np.where(num_regions >= min_regions)[0]]

    # Convert to dataframe if necessary
    if as_dataframe:
        trajectories = pd.DataFrame(data={
                            'subjects': [i['session']['subject'] for i in trajectories],
                            'dates': [i['session']['start_time'][:10] for i in trajectories],
                            'probes': [i['probe_name'] for i in trajectories],
                            'lab': [i['session']['lab'] for i in trajectories],
                            'eid': [i['session']['id'] for i in trajectories],
                            'probe_insertion': [i['probe_insertion'] for i in trajectories]})
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


def figure_style(return_colors=False):
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 4,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if return_colors:
        return {'PPC': sns.color_palette('colorblind')[0],
                'CA1': sns.color_palette('colorblind')[2],
                'DG': sns.color_palette('muted')[2],
                'LP': sns.color_palette('colorblind')[4],
                'PO': sns.color_palette('colorblind')[6]}


def combine_regions(regions):
    """
    Combine all layers of cortex and the dentate gyrus molecular and granular layer
    Combine VISa and VISam into PPC
    """
    remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
    for i, region in enumerate(regions):
        if region[:2] == 'CA':
            continue
        if (region == 'DG-mo') or (region == 'DG-sg') or (region == 'DG-po'):
            regions[i] = 'DG'
        for j, char in enumerate(remove):
            regions[i] = regions[i].replace(char, '')
        if (regions[i] == 'VISa') | (regions[i] == 'VISam'):
            regions[i] = 'PPC'
    return regions


def exclude_recordings(df, max_ap_rms=50, min_regions=3, min_channels_region=5,
                       min_neurons_per_channel=0.1, return_excluded=False):
    """
    Exclude recordings from brain regions dataframe

    Parameters
    ----------
    df : dataframe
        Dataframe with brain regions generated by the script figure3_brain_regions
    max_ap_rms : int
        Noise cutoff: maximum ap band rms to be included. The default is 50.
    min_regions : int
        Minimum amount of regions a recording must have targeted. The default is 3.
    min_channels_region : int
        Minimum number of channels needed to be in a region to count that region as sucessfully
        targeted. The default is 10.
    min_neurons_per_channel : float
        Yield cutoff: minimum neurons per channels for the entire probe
    return_excluded : bool
        Whether to also return a dataframe with the excluded recordings and why they were excluded

    Returns
    -------
    df : dataframe
        The dataframe excluding the dropped recordings.
    df_excluded : dataframe
        Dataframe with the excluded recordings and reasons (only when return_excluded=True)
    """

    # Get dataframe with excluded recordings and reason for exclusion
    df_noise = df.groupby('subject').filter(lambda s : s['rms_ap'].mean() >= max_ap_rms)
    df_noise['noise_cutoff'] = True
    df_yield = df.groupby('subject').filter(
        lambda s : (s['neuron_yield'].sum() / s['n_channels'].sum()) <= min_neurons_per_channel)
    df_yield['yield_cutoff'] = True
    df['region_hit'] = df['n_channels'] > min_channels_region
    df_target = df.groupby('subject').filter(lambda s : s['region_hit'].sum() <= min_regions)
    df_target['missed_target'] = True
    df_excluded = pd.concat((df_noise, df_yield, df_target))
    df_excluded.loc[df_excluded['noise_cutoff'].isnull(), 'noise_cutoff'] = False
    df_excluded.loc[df_excluded['yield_cutoff'].isnull(), 'yield_cutoff'] = False
    df_excluded.loc[df_excluded['missed_target'].isnull(), 'missed_target'] = False

    # Get dataframe with recordings to include
    df = df.groupby('subject').filter(lambda s : s['rms_ap'].mean() <= max_ap_rms)
    df = df.groupby('subject').filter(
        lambda s : (s['neuron_yield'].sum() / s['n_channels'].sum()) >= min_neurons_per_channel)
    df['region_hit'] = df['n_channels'] > min_channels_region
    df = df.groupby('subject').filter(lambda s : s['region_hit'].sum() >= min_regions)
    if return_excluded == False:
        return df
    else:
        return df, df_excluded


def eid_list():
    """
    Static list of eids which pass the ADVANCED criterium to include for repeated site analysis
    """
    eids = np.load('repeated_site_eids.npy')
    return eids


def eid_list_all():
    """
    Static list of all repeated site eids
    """
    eids = np.load('all_repeated_site_eids.npy')
    return eids
