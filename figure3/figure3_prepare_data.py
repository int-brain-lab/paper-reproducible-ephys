import numpy as np
import pandas as pd

from one.api import ONE
from ibllib.atlas import AllenAtlas
from ibllib.pipes.ephys_alignment import EphysAlignment
from iblutil.numerical import ismember
from brainbox.processing import compute_cluster_average
from brainbox.io.one import SpikeSortingLoader

from reproducible_ephys_functions import get_insertions, combine_regions, BRAIN_REGIONS, save_data_path, save_dataset_info
from reproducible_ephys_processing import compute_new_label
from figure3.figure3_load_data import load_dataframe


ba = AllenAtlas()

LFP_BAND = [20, 80]


def prepare_data(insertions, one, recompute=False, new_metrics=True):

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

        print(f'processing {iIns + 1}/{len(insertions)}')
        data_clust = {}
        data_chns = {}

        eid = ins['session']['id']
        lab = ins['session']['lab']
        subject = ins['session']['subject']
        date = ins['session']['start_time'][:10]
        pid = ins['probe_insertion']
        probe = ins['probe_name']

        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])

        channels['rawInd'] = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=sl.collection)
        clusters = sl.merge_clusters(spikes, clusters, channels)

        if new_metrics:
            try:
                clusters['label'] = np.load(sl.files['clusters'][0].parent.joinpath('clusters.new_labels.npy'))
            except FileNotFoundError:
                new_labels = compute_new_label(spikes, clusters, save_path=sl.files['spikes'][0].parent)
                clusters['label'] = new_labels

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
        lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
        freqs = ((lfp['freqs'] > LFP_BAND[0])
                 & (lfp['freqs'] < LFP_BAND[1]))
        power = lfp['power'][:, channels['rawInd']]
        lfp_power = np.nanmean(10 * np.log(power[freqs]), axis=0)

        data_chns['x'] = channels['x']
        data_chns['y'] = channels['y']
        data_chns['z'] = channels['z']
        data_chns['axial_um'] = channels['axial_um']
        data_chns['lateral_um'] = channels['lateral_um']
        data_chns['lfp'] = lfp_power
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

        # Data for insertion dataframe
        rms_ap = one.load_object(eid, 'ephysTimeRmsAP', collection=f'raw_ephys_data/{probe}',
                                 attribute=['rms'])
        rms_ap_data = rms_ap['rms'] * 1e6  # convert to uV
        median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
        rms_ap_data_median = (np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data)
                              + median)

        try:
            for region in BRAIN_REGIONS:
                region_clusters = np.where(np.bitwise_and(clusters['rep_site_acronym'] == region,
                                                          clusters['label'] == 1))[0]
                region_chan = channels['rawInd'][np.where(channels['rep_site_acronym'] == region)[0]]

                # Get AP band rms
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

        except Exception as err:
            print(err)

    concat_df_clust = pd.concat(all_df_clust, ignore_index=True)
    concat_df_chns = pd.concat(all_df_chns, ignore_index=True)
    save_path = save_data_path(figure='figure3')
    print(f'Saving data to {save_path}')
    concat_df_clust.to_csv(save_path.joinpath('figure3_dataframe_clust.csv'))
    concat_df_chns.to_csv(save_path.joinpath('figure3_dataframe_chns.csv'))
    metrics.to_csv(save_path.joinpath('figure3_dataframe_ins.csv'))

    return all_df_chns, all_df_clust, metrics


if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True
    insertions = get_insertions(level=0, one=one, freeze=None)
    all_df_chns, all_df_clust, metrics = prepare_data(insertions, one=one)
    save_dataset_info(one, figure='figure3')
