import numpy as np
import pandas as pd
from os.path import isfile
import traceback

import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from one.api import ONE
from ibllib.atlas import AllenAtlas
from ibllib.pipes.ephys_alignment import EphysAlignment
from iblutil.numerical import ismember
from brainbox.processing import compute_cluster_average
from brainbox.io.one import SpikeSortingLoader

import ephys_atlas.rawephys

from reproducible_ephys_functions import (get_insertions, combine_regions, BRAIN_REGIONS, save_data_path,
                                          save_dataset_info, filter_recordings)
from figure3.figure3_load_data import load_dataframe


ba = AllenAtlas()

LFP_BAND = [49, 61]
THETA_BAND = [6, 12]


def prepare_data(insertions, one, recompute=False):

    if not recompute:
        data_clust = load_dataframe(df_name='clust', exists_only=True)
        data_chn = load_dataframe(df_name='chns', exists_only=True)
        data_ins = load_dataframe(df_name='ins', exists_only=True)
        if data_clust and data_chn and data_ins:
            df = load_dataframe(df_name='ins')
            pids = np.array([p['probe_insertion'] for p in insertions])
            isin, _ = ismember(pids, df['pid'].unique())
            if np.all(isin):
                print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
                df_clust = load_dataframe(df_name='clust')
                df_chns = load_dataframe(df_name='chns')
                return df_chns, df_clust, df

    all_df_clust = []
    all_df_chns = []
    metrics = pd.DataFrame()

    for iIns, ins in enumerate(insertions):
        print(f'Processing recording {iIns + 1} of {len(insertions)}')
        data_clust, data_chns = {}, {}
        eid = ins['session']['id']
        lab = ins['session']['lab']
        subject = ins['session']['subject']
        date = ins['session']['start_time'][:10]
        pid = ins['probe_insertion']
        probe = ins['probe_name']

        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        try:
            spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
        except Exception:
            print(traceback.format_exc())
            continue

        channels['rawInd'] = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=sl.collection)
        clusters = sl.merge_clusters(spikes, clusters, channels)

        channels['rep_site_acronym'] = combine_regions(channels['acronym'])
        channels['rep_site_acronym_alt'] = np.copy(channels['rep_site_acronym'])
        channels['rep_site_acronym_alt'][channels['rep_site_acronym_alt'] == 'PPC'] = 'VISa'
        channels['rep_site_id'] = ba.regions.acronym2id(channels['rep_site_acronym_alt'])
        clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])

        # Data for cluster dataframe
        cluster_idx = np.where(clusters['label'] == 1)
        data_clust['cluster_ids'] = clusters['cluster_id'][cluster_idx]
        data_clust['region'] = clusters['rep_site_acronym'][cluster_idx]
        # Find spikes that are from the clusterIDs
        spike_idx = np.isin(spikes['clusters'], data_clust['cluster_ids'])

        clu, data_clust['depths'], n_cluster = compute_cluster_average(spikes['clusters'][spike_idx], spikes['depths'][spike_idx])
        assert np.array_equal(clu, data_clust['cluster_ids'])
        clu, data_clust['amps'], _ = compute_cluster_average(spikes['clusters'][spike_idx], spikes['amps'][spike_idx])
        assert np.array_equal(clu, data_clust['cluster_ids'])
        data_clust['fr'] = n_cluster / np.max(spikes.times) - np.min(spikes.times)

        insertion = one.alyx.rest('insertions', 'list', id=pid)[0]
        traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track', probe_insertion=pid)[0]
        feature, track = [*traj['json'][insertion['json']['extended_qc']['alignment_stored']]][:2]
        xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6
        ephysalign = EphysAlignment(xyz_picks, channels['axial_um'], brain_atlas=ba, feature_prev=feature, track_prev=track,
                                    speedy=True)
        data_clust['depths_aligned'] = ephysalign.get_channel_locations(feature, track, data_clust['depths'] / 1e6)[:, 2] * 1e6

        df_clust = pd.DataFrame.from_dict(data_clust)
        df_clust['eid'] = eid
        df_clust['pid'] = pid
        df_clust['subject'] = ins['session']['subject']
        df_clust['probe'] = ins['probe_name']
        df_clust['date'] = ins['session']['start_time'][:10]
        df_clust['lab'] = ins['session']['lab']

        # Data for channel dataframe
        try:
            pid_directory = save_data_path().joinpath('lfp_destripe_snippets').joinpath(ins['id'])
            # this function will download a set of 30 seconds examples and destripe them
            ephys_atlas.rawephys.destripe(pid, one=one, typ='lf', prefix="", destination=pid_directory, remove_cached=True, clobber=False)
            # then we loop over the snippets and compute the RMS of each
            lfp_files = list(pid_directory.rglob('lf.npy'))
            for j, lfp_file in enumerate(lfp_files):
                lfp = np.load(lfp_file).astype(np.float32)
                f, pow = scipy.signal.periodogram(lfp, fs=250, scaling='density')
                if j == 0:
                    rms_lf_band, rms_lf = (np.zeros((lfp.shape[0], len(lfp_files))) for i in range(2))
                rms_lf_band[:, j] = np.nanmean(10 * np.log10(pow[:, np.logical_and(f >= LFP_BAND[0], f <= LFP_BAND[1])]), axis=-1)
                rms_lf[:, j] = np.mean(np.sqrt(lfp.astype(np.double) ** 2), axis=-1)
            lfp_power = np.nanmedian(rms_lf_band - 20 * np.log10(f[1]), axis=-1) * 2
            lfp_rms = np.median(20 * np.log10(rms_lf), axis=-1) * 2
        except Exception:
            print(f'pid: {pid} RAW LFP ERROR \n', traceback.format_exc())
            lfp_power = np.nan
            lfp_rms = np.nan

        try:
            lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
            # Get broadband lfp power
            freqs = (lfp['freqs'] >= LFP_BAND[0]) & (lfp['freqs'] <= LFP_BAND[1])
            power = lfp['power'][:, channels['rawInd']]
            lfp_power_raw = np.nanmean(10 * np.log(power[freqs]), axis=0)
            # # Get theta band lfp power
            # freqs = (lfp['freqs'] >= THETA_BAND[0]) & (lfp['freqs'] <= THETA_BAND[1])
            # power = lfp['power'][:, channels['rawInd']]
            # theta_power = np.nanmean(10 * np.log(power[freqs]), axis=0)
        except Exception:
            print(f'pid: {pid} LFP ERROR \n', traceback.format_exc())
            lfp_power_raw = np.nan

        data_chns['x'] = channels['x']
        data_chns['y'] = channels['y']
        data_chns['z'] = channels['z']
        data_chns['axial_um'] = channels['axial_um']
        data_chns['lateral_um'] = channels['lateral_um']
        data_chns['lfp'] = lfp_power
        data_chns['lfp_rms'] = lfp_rms
        data_chns['lfp_raw'] = lfp_power_raw
        data_chns['region_id'] = channels['atlas_id']
        data_chns['region_id_rep'] = channels['rep_site_id']
        data_chns['region'] = channels['rep_site_acronym']

        df_chns = pd.DataFrame.from_dict(data_chns)
        df_chns['eid'] = eid
        df_chns['pid'] = pid
        df_chns['subject'] = ins['session']['subject']
        df_chns['probe'] = ins['probe_name']
        df_chns['date'] = ins['session']['start_time'][:10]
        df_chns['lab'] = ins['session']['lab']
        all_df_clust.append(df_clust)
        all_df_chns.append(df_chns)

        try:
            # Data for insertion dataframe
            rms_ap = one.load_object(eid, 'ephysTimeRmsAP', collection=f'raw_ephys_data/{probe}', attribute=['rms'])
            rms_ap_data = rms_ap['rms'] * 1e6 if np.mean(rms_ap['rms']) < 0.1 else rms_ap['rms']
            median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
            rms_ap_data_median = (np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data) + median)
        except BaseException:
            print(f'pid: {pid} AP ERROR\n', traceback.format_exc())
            rms_ap_data_median = np.nan

        for region in BRAIN_REGIONS:
            region_clusters = np.where(np.bitwise_and(clusters['rep_site_acronym'] == region,
                                                      clusters['label'] == 1))[0]
            region_chan = channels['rawInd'][np.where(channels['rep_site_acronym'] == region)[0]]

            # Get AP band rms
            if np.isnan(rms_ap_data_median).any():
                rms_ap_region = np.nan
            else:
                rms_ap_region = np.median(rms_ap_data_median[:, region_chan])

            if region_clusters.size == 0:
                metrics = pd.concat((metrics, pd.DataFrame(
                    index=[metrics.shape[0] + 1], data={'pid': pid, 'eid': eid, 'probe': probe,
                                                        'lab': lab, 'subject': subject,
                                                        'region': region, 'date': date,
                                                        'median_firing_rate': np.nan,
                                                        'mean_firing_rate': np.nan,
                                                        'spike_amp_mean': np.nan,
                                                        'spike_amp_median': np.nan,
                                                        'spike_amp_90': np.nan,
                                                        'rms_ap': rms_ap_region})))
            else:
                # Get firing rate and spike amplitude
                neuron_fr = np.empty(len(region_clusters))
                spike_amp = np.empty(len(region_clusters))
                for n, neuron_id in enumerate(region_clusters):
                    neuron_fr[n] = np.sum(spikes['clusters'] == neuron_id) / np.max(spikes['times'])
                    spike_amp[n] = np.median(spikes.amps[spikes['clusters'] == neuron_id])

                # Add to dataframe
                metrics = pd.concat((metrics, pd.DataFrame(
                    index=[metrics.shape[0] + 1], data={'pid': pid, 'eid': eid, 'probe': probe,
                                                        'lab': lab, 'subject': subject,
                                                        'region': region, 'date': date,
                                                        'median_firing_rate': np.median(neuron_fr),
                                                        'mean_firing_rate': np.mean(neuron_fr),
                                                        'spike_amp_mean': np.nanmean(spike_amp) * 1e6,
                                                        'spike_amp_median': np.nanmedian(spike_amp),
                                                        'spike_amp_90': np.percentile(spike_amp, 95),
                                                        'rms_ap': rms_ap_region})))

    concat_df_clust = pd.concat(all_df_clust, ignore_index=True)
    concat_df_chns = pd.concat(all_df_chns, ignore_index=True)
    save_path = save_data_path(figure='figure3')
    print(f'Saving data to {save_path}')
    concat_df_clust.to_csv(save_path.joinpath('figure3_dataframe_clust.csv'))
    concat_df_chns.to_csv(save_path.joinpath('figure3_dataframe_chns.csv'))
    metrics.to_csv(save_path.joinpath('figure3_dataframe_ins.csv'))

    return all_df_chns, all_df_clust, metrics


def run_decoding(metrics=['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_mean'],
                 qc='pass', n_shuffle=500, min_lab_region=2, recompute=False):
    """
    qc can be "pass" (only include recordings that pass QC)
    "high_noise": add the recordings with high noise
    "low_yield": add low yield recordings
    "missed_target": add recordings that missed the target regions
    "artifacts": add recordings with artifacts
    "low_trials": add recordings with < 400 trials
    "high_lfp": add recordings with high LFP power
    "all": add all recordings regardless of QC
    """
    save_path = save_data_path(figure='figure3')
    file_name = f'figure3_dataframe_decode_{qc}.csv'
    if recompute or not isfile(save_path.joinpath(file_name)):

        # Initialize
        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        kfold = KFold(n_splits=5, shuffle=False)

        # Load in data
        df_ins = load_dataframe(df_name='ins')
        data = filter_recordings(df_ins, min_lab_region=min_lab_region, min_rec_lab=0,
                                 min_neuron_region=2)
        if qc == 'pass':
            data = data[data['include'] == 1]  # select recordings that pass QC
        elif qc != 'all':
            data = data[(data['include'] == 1) | (data[qc] == 1)]

        data = data[data['lfp_power'].notna()]  # exclude recordings that miss LFP data
        data['yield_per_channel'] = data['neuron_yield'] / data['n_channels']

        # Restructure dataframe
        data.loc[data['region'] == 'PPC', 'region_number'] = 1
        data.loc[data['region'] == 'CA1', 'region_number'] = 2
        data.loc[data['region'] == 'DG', 'region_number'] = 3
        data.loc[data['region'] == 'LP', 'region_number'] = 4
        data.loc[data['region'] == 'PO', 'region_number'] = 5
        data = data[~data['median_firing_rate'].isnull()]

        # Decode per brain region
        decode_df, shuffle_df = pd.DataFrame(), pd.DataFrame()
        for r in ['PPC', 'CA1', 'DG', 'LP', 'PO']:
            print(f'\nDecoding lab for region {r}..\n')
            region_data = data[data['region'] == r]
            decode_data = region_data[metrics].to_numpy()
            decode_labs = region_data['institute'].values

            # Decode lab
            lab_predict = np.empty(decode_data.shape[0]).astype(object)
            for train_index, test_index in kfold.split(decode_data):
                rf.fit(decode_data[train_index], decode_labs[train_index])
                lab_predict[test_index] = rf.predict(decode_data[test_index])

            # Get confusion matrix
            matrix = confusion_matrix(decode_labs, lab_predict, labels=np.unique(decode_labs))
            matrix.diagonal()/matrix.sum(axis=1)
            matrix_df = pd.DataFrame(data=matrix, index=np.unique(decode_labs),
                                     columns=np.unique(decode_labs))

            # Save results in dataframe
            dict_df = dict(zip(metrics, rf.feature_importances_))
            dict_df['region'] = r
            dict_df['n_labs'] = np.unique(decode_labs).shape[0]
            dict_df['accuracy'] = accuracy_score(decode_labs, lab_predict)
            decode_df = pd.concat((decode_df, pd.DataFrame(index=[decode_df.shape[0]+1], data=dict_df)))

            # Decode lab with shuffled lab labels
            shuf_acc = np.empty(n_shuffle)
            for i in range(n_shuffle):
                if np.mod(i, 100) == 0:
                    print(f'Shuffle {i} of {n_shuffle}')
                lab_shuf = shuffle(decode_labs)
                lab_predict = np.empty(decode_data.shape[0]).astype(object)
                for train_index, test_index in kfold.split(decode_data):
                    rf.fit(decode_data[train_index], lab_shuf[train_index])
                    lab_predict[test_index] = rf.predict(decode_data[test_index])
                shuf_acc[i] = accuracy_score(decode_labs, lab_predict)

            shuffle_df = pd.concat((shuffle_df, pd.DataFrame(data={
                'region': r, 'accuracy_shuffle': shuf_acc})), ignore_index=True)

        # Decode region
        print('\nDecoding brain region..\n')
        decode_data = data[['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap',
                            'spike_amp_mean']].to_numpy()
        decode_regions = data['region'].values
        region_predict = np.empty(data.shape[0]).astype(object)
        for train_index, test_index in kfold.split(decode_data):
            rf.fit(decode_data[train_index], decode_regions[train_index])
            region_predict[test_index] = rf.predict(decode_data[test_index])
        decode_df = pd.concat((decode_df, pd.DataFrame(index=[decode_df.shape[0]+1], data={
            'region': 'all', 'accuracy': accuracy_score(decode_regions, region_predict)})))

        # Decode lab with shuffled lab labels
        shuf_acc = np.empty(n_shuffle)
        for i in range(n_shuffle):
            if np.mod(i, 100) == 0:
                print(f'Shuffle {i} of {n_shuffle}')
            region_shuf = shuffle(decode_regions)
            region_predict = np.empty(data.shape[0]).astype(object)
            for train_index, test_index in kfold.split(decode_data):
                rf.fit(decode_data[train_index], region_shuf[train_index])
                region_predict[test_index] = rf.predict(decode_data[test_index])
            shuf_acc[i] = accuracy_score(decode_regions, region_predict)

        shuffle_df = pd.concat((shuffle_df, pd.DataFrame(data={
            'region': 'all', 'accuracy_shuffle': shuf_acc})), ignore_index=True)

        # Save results
        decode_df.to_csv(save_path.joinpath(file_name))
        shuffle_df.to_csv(save_path.joinpath(f'figure3_dataframe_decode_shuf_{qc}.csv'))
        matrix_df.to_csv(save_path.joinpath(f'figure3_dataframe_conf_mat_{qc}.csv'))

    else:
        print('Decoding results found, not running again')


if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True
    insertions = get_insertions(level=0, one=one, freeze='release_2022_11')
    all_df_chns, all_df_clust, metrics = prepare_data(insertions, recompute=True, one=one)
    save_dataset_info(one, figure='figure3')
    rerun_decoding = False
    run_decoding(n_shuffle=500, qc='pass', recompute=rerun_decoding)
    run_decoding(n_shuffle=500, qc='high_noise', recompute=rerun_decoding)
    run_decoding(n_shuffle=500, qc='low_yield', recompute=rerun_decoding)
    run_decoding(n_shuffle=500, qc='high_lfp', recompute=rerun_decoding)
    run_decoding(n_shuffle=500, qc='artifacts', recompute=rerun_decoding)
    run_decoding(n_shuffle=500, qc='missed_target', recompute=rerun_decoding)
    run_decoding(n_shuffle=500, qc='low_trials', recompute=rerun_decoding)
    run_decoding(n_shuffle=500, qc='all', recompute=rerun_decoding)
