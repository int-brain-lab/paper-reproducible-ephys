"""
General functions for reproducible ephys paper
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import logging
from pathlib import Path

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
from iblutil.numerical import ismember
from ibllib.atlas import AllenAtlas
import brainbox.io.one as bbone
from brainbox.metrics.single_units import quick_unit_metrics
from brainbox.behavior import training

from one.params import get_cache_dir

logger = logging.getLogger('paper_reproducible_ephys')


STR_QUERY = 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,' \
            'probe_insertion__session__qc__lt,50,' \
            '~probe_insertion__json__qc,CRITICAL,' \
            'probe_insertion__session__n_trials__gte,400'

BRAIN_REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']


def labs():
    lab_number_map = {'cortexlab': 'Lab 1', 'mainenlab': 'Lab 2', 'churchlandlab': 'Lab 3',
                      'angelakilab': 'Lab 4', 'wittenlab': 'Lab 5', 'hoferlab': 'Lab 6',
                      'mrsicflogellab': 'Lab 6', 'danlab': 'Lab 7', 'zadorlab': 'Lab 8',
                      'steinmetzlab': 'Lab 9', 'churchlandlab_ucla': 'Lab 10'}
    institution_map = {'cortexlab': 'UCL', 'mainenlab': 'CCU', 'zadorlab': 'CSHL (Z)',
                       'churchlandlab': 'CSHL (C)', 'angelakilab': 'NYU',
                       'wittenlab': 'Princeton', 'hoferlab': 'SWC', 'mrsicflogellab': 'SWC',
                       'danlab': 'Berkeley', 'steinmetzlab': 'UW', 'churchlandlab_ucla': 'UCLA'}
    colors = np.concatenate([sns.color_palette("Dark2"), sns.color_palette('Set2')[0:2]])
    institutions = ['UCL', 'CCU', 'CSHL (C)', 'NYU', 'Princeton', 'SWC', 'Berkeley', 'CSHL (Z)',
                    'UW', 'UCLA']
    institution_colors = {}
    for i, inst in enumerate(institutions):
        institution_colors[inst] = colors[i]
    return lab_number_map, institution_map, institution_colors


def query(behavior=False, n_trials=400, resolved=True, min_regions=2, exclude_critical=True, one=None, str_query=None,
          as_dataframe=False):

    one = one or ONE()

    if isinstance(str_query, str):
        django_queries = [str_query]
    elif isinstance(str_query, list):
        django_queries = str_query
    else:
        django_queries = []

    if exclude_critical:
        django_queries.append('probe_insertion__session__qc__lt,50,~probe_insertion__json__qc,CRITICAL')
    if resolved:
        django_queries.append('probe_insertion__json__extended_qc__alignment_resolved,True')
    if behavior:
        django_queries.append('probe_insertion__session__extended_qc__behavior,1')
    if n_trials > 0:
        django_queries.append(f'probe_insertion__session__n_trials__gte,{n_trials}')

    django_query = ','.join(django_queries)

    trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                 x=-2243, y=-2000, theta=15, project='ibl_neuropixel_brainwide_01',
                                 django=django_query)
    pids = [traj['probe_insertion'] for traj in trajectories]

    if min_regions > 0:
        region_traj = []
        query_regions = BRAIN_REGIONS.copy()
        query_regions[query_regions.index('PPC')] = 'VIS'
        # Query how many of the target regions were hit per recording
        for i, region in enumerate(query_regions):
            region_query = one.alyx.rest(
                'trajectories', 'list', provenance='Ephys aligned histology track',
                django=f'{django_query},channels__brain_region__acronym__icontains,{region},probe_insertion__in,{pids}')
            region_traj = np.append(region_traj, [i['probe_insertion'] for i in region_query])

        region_pids, num_regions = np.unique(region_traj, return_counts=True)
        isin, _ = ismember(np.array(pids), region_pids[num_regions >= min_regions])
        trajectories = [trajectories[i] for i, val in enumerate(isin) if val]

    # Convert to dataframe if necessary
    if as_dataframe:
        trajectories = pd.DataFrame(data={
            'subjects': [i['session']['subject'] for i in trajectories],
            'dates': [i['session']['start_time'][:10] for i in trajectories],
            'probes': [i['probe_name'] for i in trajectories],
            'lab': [i['session']['lab'] for i in trajectories],
            'eid': [i['session']['id'] for i in trajectories],
            'probe_insertion': [i['probe_insertion'] for i in trajectories]})
        trajectories['institution'] = trajectories.lab.map(labs()[1])
    return trajectories


def get_insertions(level=2, recompute=False, as_dataframe=False, one=None):
    """
    Find insertions used for analysis based on different exclusion levels
    Level 0: minimum_regions = 0, resolved = True, behavior = False, n_trial >= 0, exclude_critical = True
    Level 1: minimum_regions = 2, resolved = True, behavior = False, n_trial >= 400, exclude_critical = True
    Level 2: minimum_regions = 2, resolved = True, behavior = False, n_trial >= 400, exclude_critical = True,
             max_ap_rms=40,  max_lfp_power=-140, min_channels_per_region=5, min_neurons_per_channel=0.1

    :param level: exclusion level 0, 1 or 2
    :param recompute: whether to recompute the metrics dataframe that is used to exclude recordings at level=2
    :param as_dataframe: whether to return a dict or a dataframe
    :param one: ONE instance
    :return: dict or pandas dataframe of probe insertions
    """
    one = one or ONE()
    if level == 0:
        insertions = query(min_regions=0, n_trials=0, behavior=False, exclude_critical=True, one=one, as_dataframe=as_dataframe)
        if recompute:
            _ = recompute_metrics(insertions, one)
        return insertions

    if level == 1:
        insertions = query(one=one, as_dataframe=as_dataframe)
        if recompute:
            ins = query(min_regions=0, n_trials=0, behavior=False, exclude_critical=True, one=one, as_dataframe=True)
            _ = recompute_metrics(ins, one)
        return insertions

    if level >= 2:
        insertions = query(one=one, as_dataframe=False)
        pids = np.array([ins['probe_insertion'] for ins in insertions])
        ins = filter_recordings(min_neuron_region=0)
        ins = ins[ins['include']]
        if recompute:
            _ = recompute_metrics(insertions, one)
            isin, _ = ismember(pids, ins['pid'].unique())
            ins = [insertions[i] for i, val in enumerate(isin) if val]

        return ins


def recompute_metrics(insertions, one):
    """
    Determine whether metrics need to be recomputed or not for given list of insertions
    :param insertions: list of insertions
    :param one: ONE object
    :return:
    """

    pids = np.array([ins['probe_insertion'] for ins in insertions])
    metrics = load_metrics()
    if (metrics is None) or (metrics.shape[0] == 0):
        metrics = compute_metrics(insertions, one=one)
    else:
        isin, _ = ismember(pids, metrics['pid'].unique())
        if not np.all(isin):
            metrics = compute_metrics(insertions, one=one)

    return metrics


def query_histology():
    '''
    Return the trajectories for histology analysis

    Returns
    -------
    List of histology subjs

    '''
    import figure_hist_data as fhd

    data = fhd.get_probe_data()

    return list(data['subject'])



def data_path():
    # Retrieve absolute path of paper-behavior dir
    repo_dir = Path(__file__).resolve().parent
    data_dir = repo_dir.joinpath('data')
    data_dir.mkdir(exist_ok=True)

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
                'PO': sns.color_palette('colorblind')[6],
                'RS': sns.color_palette('Set2')[0],
                'FS': sns.color_palette('Set2')[1],
                'RS1': sns.color_palette('Set2')[2],
                'RS2': sns.color_palette('Set2')[3]}


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


def exclude_recordings(df, max_ap_rms=40, min_regions=3, min_channels_region=5, max_lfp_power=-140,
                       min_neurons_per_channel=0.1,
                       return_excluded=False, destriped_rms=True):
    """
    Exclude recordings from brain regions dataframe

    Parameters
    ----------
    df : dataframe
        Dataframe with brain regions generated by the script figure3_brain_regions
    max_ap_rms : int
        Noise cutoff: maximum destriped ap band rms (90th perc) to be included. The default is 40.
    min_regions : int
        Minimum amount of regions a recording must have targeted. The default is 3.
    min_channels_region : int
        Minimum number of channels needed to be in a region to count that region as sucessfully
        targeted. The default is 10.
    min_neurons_per_channel : float
        Yield cutoff: minimum neurons per channels for the entire probe
    return_excluded : bool
        Whether to also return a dataframe with the excluded recordings and why they were excluded
    destriped_rms : bool
        Whether to use destriped RMS or regular RMS for the noise cutoff

    Returns
    -------
    df : dataframe
        The dataframe excluding the dropped recordings.
    df_excluded : dataframe
        Dataframe with the excluded recordings and reasons (only when return_excluded=True)
    """

    # If destriped rms is missing, set to 0
    df.loc[df['rms_ap_p90'].isnull(), 'rms_ap_p90'] = 0

    # Get dataframe with excluded recordings and reason for exclusion
    df_excluded = pd.DataFrame()
    if destriped_rms:
        df_excluded['high_noise'] = df.groupby('subject')['rms_ap_p90'].median() >= max_ap_rms
    else:
        df_excluded['high_noise'] = df.groupby('subject')['rms_ap'].median() >= max_ap_rms

    df_excluded['high_lfp'] = df.groupby('subject')['lfp_power_high'].median() >= max_lfp_power
    df_excluded['low_yield'] = (
                    df.groupby('subject')['neuron_yield'].sum()
                    / df.groupby('subject')['n_channels'].sum()) < min_neurons_per_channel
    df['region_hit'] = df['n_channels'] > min_channels_region
    df_excluded['missed_target'] = df.groupby('subject')['region_hit'].sum() < min_regions
    df_excluded['artifacts'] = False
    df_excluded.loc['DY_013', 'artifacts'] = True
    df_excluded.loc['ibl_witten_26', 'artifacts'] = True
    df_excluded['excluded'] = df_excluded.any(axis=1, bool_only=True)
    df_excluded = df_excluded.reset_index()

    # Get dataframe with recordings to include
    df = df[df['subject'] != 'DY_013']  # exclude mouse DY_013
    df = df[df['subject'] != 'ibl_witten_26']  # exclude mouse ibl_witten_26
    if destriped_rms:
        df = df.groupby('subject').filter(lambda s : s['rms_ap_p90'].median() <= max_ap_rms)
    else:
        df = df.groupby('subject').filter(lambda s : s['rms_ap'].median() <= max_ap_rms)
    df = df.groupby('subject').filter(lambda s : s['lfp_power_high'].median() <= max_lfp_power)
    df = df.groupby('subject').filter(
        lambda s : (s['neuron_yield'].sum() / s['n_channels'].sum()) >= min_neurons_per_channel)
    df['region_hit'] = df['n_channels'] >= min_channels_region
    df = df.groupby('subject').filter(lambda s : s['region_hit'].sum() >= min_regions)

    if return_excluded == False:
        return df
    else:
        return df, df_excluded

def eid_list():
    """
    Static list of eids which pass the ADVANCED criterium to include for repeated site analysis
    """
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    eids = np.load(os.path.join(repo_dir, 'repeated_site_eids.npy'))
    return eids


def eid_list_all():
    """
    Static list of all repeated site eids
    """
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    eids = np.load(os.path.join(repo_dir, 'all_repeated_site_eids.npy'))
    return eids

def pid_list():
    """
    Static list of all repeated site eids
    """
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    pids = np.load(os.path.join(repo_dir, 'repeated_site_pids.npy'))
    return pids


def load_metrics():
    if save_data_path().joinpath('insertion_metrics.csv').exists():
        metrics = pd.read_csv(save_data_path().joinpath('insertion_metrics.csv'))
    else:
        metrics = None
    return metrics


def save_data_path(figure=None):
    """
    Path to save data
    :param figure: figure number e.g 'figure3'
    :return:
    """
    try:
        import reproducible_ephys_paths
        defined_paths = dir(reproducible_ephys_paths)
        if 'DATA_PATH' in defined_paths:
            data_path = Path(reproducible_ephys_paths.DATA_PATH)
        else:
            data_path = Path(get_cache_dir()).joinpath('paper_repro_ephys')

    except ModuleNotFoundError:
        data_path = Path(get_cache_dir()).joinpath('paper_repro_ephys')

    if figure is not None:
        data_path = data_path.joinpath(str(figure))

    data_path.mkdir(exist_ok=True, parents=True)

    return data_path


def save_figure_path(figure=None):
    """
    Path to save figures
    :param figure: figure number e.g 'figure3'
    :return:
    """
    try:
        import reproducible_ephys_paths
        defined_paths = dir(reproducible_ephys_paths)
        if 'FIG_PATH' in defined_paths:
            fig_path = Path(reproducible_ephys_paths.FIG_PATH)
        else:
            fig_path = Path(get_cache_dir()).joinpath('paper_repro_ephys')

    except ModuleNotFoundError:
        fig_path = Path(get_cache_dir()).joinpath('paper_repro_ephys')

    if figure is not None:
        fig_path = fig_path.joinpath(str(figure), 'figures')
    else:
        fig_path = fig_path.joinpath('figures')

    fig_path.mkdir(exist_ok=True, parents=True)

    return fig_path


def compute_metrics(insertions, one=None, ba=None, spike_sorter='pykilosort', save=True):
    one = one or ONE()
    ba = ba or AllenAtlas()
    lab_number_map, institution_map, _ = labs()
    metrics = pd.DataFrame()
    LFP_BAND = [20, 80]

    for i, ins in enumerate(insertions):
        eid = ins['session']['id']
        lab = ins['session']['lab']
        subject = ins['session']['subject']
        date = ins['session']['start_time'][:10]
        pid = ins['probe_insertion']
        probe = ins['probe_name']
        collection = f'alf/{probe}/{spike_sorter}'

        try:
            trials = one.load_object(eid, 'trials', collection='alf', attribute=training.TRIALS_KEYS)
            n_trials = trials["stimOn_times"].shape[0]
            behav = training.criterion_delay(n_trials=n_trials, perf_easy=training.compute_performance_easy(trials)).astype(bool)
        except Exception as err:
            sess_details = one.alyx.rest('sessions', 'read', id=eid)
            n_trials = sess_details['n_trials']
            behav = bool(sess_details['extended_qc']['behavior'])

        try:
            ap = one.load_object(eid, 'ephysChannels', collection=f'raw_ephys_data/{probe}', attribute=['apRMS'])
            if 'apRMS' not in ap.keys():
                pid_qc = one.alyx.rest('insertions', 'list', id=pid)[0]['json']['extended_qc']
                ap_val = pid_qc['apRms_p90_proc'] * 1e6 if 'apRms_p90_proc' in pid_qc.keys() else 0
        except ALFObjectNotFound:
            logger.warning(f'ephysChannels object not found for pid: {pid}')
            ap = {}
            try:
                pid_qc = one.alyx.rest('insertions', 'list', id=pid)[0]['json']['extended_qc']
                ap_val = pid_qc['apRms_p90_proc'] * 1e6 if 'apRms_p90_proc' in pid_qc.keys() else 0
            except Exception:
                ap_val = np.nan

        try:
            lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
        except ALFObjectNotFound:
            logger.warning(f'ephysSpectralDensityLF object not found for pid: {pid}')
            lfp = {}

        try:
            clusters = one.load_object(eid, 'clusters', collection=collection, attribute=['metrics', 'channels'])
            if 'metrics' not in clusters.keys():
                spikes, clusters = bbone.load_spike_sorting(eid, probe=probe, spike_sorter='pykilosort',
                                                            dataset_types=['spikes.amps', 'spikes.depths'], one=one)
                spikes = spikes[probe]
                clusters = clusters[probe]
                clusters['metrics'] = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths,
                                                         cluster_ids=np.arange(clusters.channels.size))

            channels = bbone.load_channel_locations(eid, probe=probe, one=one, aligned=True, brain_atlas=ba)[probe]
            channels['rawInd'] = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=collection)
            channels['rep_site_acronym'] = combine_regions(channels['acronym'])
            clusters['rep_site_acronym'] = channels['rep_site_acronym'][clusters['channels']]

        except Exception as err:
            print(err)

        try:
            for region in BRAIN_REGIONS:
                region_clusters = np.where(np.bitwise_and(clusters['rep_site_acronym'] == region,
                                                          clusters['metrics']['label'] == 1))[0]
                region_chan = channels['rawInd'][np.where(channels['rep_site_acronym'] == region)[0]]

                if 'power' in lfp.keys():
                    freqs = (lfp['freqs'] > LFP_BAND[0]) & (lfp['freqs'] < LFP_BAND[1])
                    chan_power = lfp['power'][:, region_chan]
                    lfp_region = np.mean(10 * np.log(chan_power[freqs]))  # convert to dB
                else:
                    # TO DO SEE IF THIS IS LEGIT
                    lfp_region = np.nan

                if 'apRMS' in ap.keys() and region_chan.shape[0] > 0:
                    ap_rms = np.percentile(ap['apRMS'][1, region_chan], 90) * 1e6
                elif region_chan.shape[0] > 0:
                    ap_rms = ap_val
                else:
                    ap_rms = 0

                metrics = pd.concat((metrics, pd.DataFrame(
                    index=[metrics.shape[0] + 1], data={'pid': pid, 'eid': eid, 'probe': probe,
                                                        'lab': lab, 'subject': subject, 'institute': institution_map[lab],
                                                        'lab_number': lab_number_map[lab],
                                                        'region': region, 'date': date,
                                                        'n_channels': region_chan.shape[0],
                                                        'neuron_yield': region_clusters.shape[0],
                                                        'lfp_power': lfp_region,
                                                        'rms_ap_p90': ap_rms,
                                                        'n_trials': n_trials,
                                                        'behavior': behav})))
        except Exception as err:
            print(err)

    if save:
        metrics.to_csv(save_data_path().joinpath('insertion_metrics.csv'))

    return metrics


def filter_recordings(df=None, max_ap_rms=40, max_lfp_power=-140, min_neurons_per_channel=0.1, min_channels_region=5,
                      min_regions=3, min_neuron_region=4, min_lab_region=3, min_rec_lab=4, n_trials=400, behavior=False,
                      exclude_subjects=['DY013', 'ibl_witten_26']):
    """
    Filter values in dataframe according to different exclusion criteria
    :param df: pandas dataframe
    :param max_ap_rms:
    :param max_lfp_power:
    :param min_neurons_per_channel:
    :param min_channels_region:
    :param min_regions:
    :param min_neuron_region:
    :param min_lab_region:
    :param n_trials:
    :param behavior:
    :return:
    """

    # Load in the insertion metrics
    metrics = load_metrics()

    if df is None:
        df = metrics
        df['original_index'] = df.index
    else:
        # make sure that all pids in the dataframe df are included in metrics 
        isin, _ = ismember(df['pid'].unique(), metrics['pid'].unique())
        print(f'Warning: {np.sum(~isin)} recordings are missing metrics')
        df = df.merge(metrics, on=['pid', 'region', 'subject', 'eid', 'probe', 'date', 'lab'])
        if 'lfp_power' not in df.keys():
            df['lfp_power'] = df['lfp_power_x']  # CHECK WITH MAYO

    # Region Level
    # no. of channels per region
    df['region_hit'] = df['n_channels'] > min_channels_region
    # no. of neurons per region
    df['low_neurons'] = df['neuron_yield'] < min_neuron_region

    # PID level
    df = df.groupby('pid').apply(lambda m: m.assign(high_noise=lambda m: m['rms_ap_p90'].median() > max_ap_rms))
    df = df.groupby('pid').apply(lambda m: m.assign(high_lfp=lambda m: m['lfp_power'].median() > max_lfp_power))
    df = df.groupby('pid').apply(lambda m: m.assign(low_yield=lambda m: (m['neuron_yield'].sum() / m['n_channels'].sum())
                                                                        < min_neurons_per_channel))
    df = df.groupby('pid').apply(lambda m: m.assign(missed_target=lambda m: m['region_hit'].sum() < min_regions))
    df = df.groupby('pid').apply(lambda m: m.assign(low_trials=lambda m: m['n_trials'] < n_trials))

    sum_metrics = df['high_noise'] + df['high_lfp'] + df['low_yield'] + df['missed_target'] + df['low_trials'] + df['low_neurons']

    if behavior:
        sum_metrics += ~df['behavior']

    df['include'] = sum_metrics == 0

    # Exclude subjects indicated to exclude from further analysis
    df['artifacts'] = bool(0)
    for subj in exclude_subjects:
        idx = df.loc[(df['subject'] == subj)].index
        df.loc[idx, 'include'] = False
        df.loc[idx, 'artifacts'] = True

    df['lab_include'] = bool(0)
    df['permute_include'] = bool(0)

    # For permutation tests
    df_red = df[df['include'] == 1]

    # Have minimum number of recordings per lab
    inst_count = df_red.groupby(['institute']).pid.nunique()
    institutes = [key for key, val in inst_count.items() if val >= min_rec_lab]

    for lab in institutes:
        idx = df.loc[(df['institute'] == lab) & df['include'] == 1].index
        df.loc[idx, 'lab_include'] = True

    # Now for permutation test
    df_red = df_red[df_red['institute'].isin(institutes)]

    # Minimum number of recordings per lab per region with enough good units
    labreg = {val: {reg: 0 for reg in df.region.unique()} for val in institutes}
    df_red = df_red.groupby(['institute', 'pid', 'region'])
    for key in df_red.groups.keys():
        df_k = df_red.get_group(key)
        # needs to be based on the neuron_yield
        if df_k.iloc[0]['neuron_yield'] > min_neuron_region:
            labreg[key[0]][key[2]] += 1

    for lab, regs in labreg.items():
        for reg, count in regs.items():
            if count >= min_lab_region:
                idx = df.loc[(df['region'] == reg) & (df['institute'] == lab) & df['include'] == 1].index
                df.loc[idx, 'permute_include'] = True

    # Sort the index so it is the same as the orignal frame that was passed in
    # df = df.sort_values('original_index').reset_index(drop=True)

    return df