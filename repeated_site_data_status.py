from one.api import ONE
from pathlib import Path


def get_repeated_site_status(subj, date, probe, one=None):

    one = one or ONE()
    eid = one.search(subject=subj, date=date)
    print(len(eid))

    dtypes_ks2 = [
        'spikes.depths',
        'spikes.clusters',
        'spikes.times',
        'clusters.depths',
        'clusters.channels',
        'clusters.amps',
        'clusters.peakToTrough',
        'clusters.waveforms',
        #'clusters.metrics',
        'channels.localCoordinates',
        'channels.rawInd'
    ]

    collections = one.list_collections(eid[0])
    # look to see if pykilosort exists
    if f'alf/{probe}/pykilosort' in collections:
        data = one.alyx.rest('datasets', 'list', subject=subj, date=date,
                             collection=f'alf/{probe}/pykilosort')
    else:
        data = one.alyx.rest('datasets', 'list', subject=subj, date=date,
                             collection=f'alf/{probe}')

    ks2_data = [str(Path(dat['name']).stem) for dat in data]

    ks2_exists = all([da in ks2_data for da in dtypes_ks2])

    data = one.alyx.rest('datasets', 'list', subject=subj, date=date,
                         collection=f'raw_ephys_data/{probe}')
    ephys_data = [str(Path(dat['name']).stem) for dat in data]
    dtypes_ephys = [
        '_iblqc_ephysTimeRmsLF.rms',
        '_iblqc_ephysTimeRmsLF.timestamps',
        '_iblqc_ephysSpectralDensityLF.power',
        '_iblqc_ephysSpectralDensityLF.freqs',
        '_iblqc_ephysTimeRmsAP.rms',
        '_iblqc_ephysTimeRmsAP.timestamps',
        '_iblqc_ephysSpectralDensityAP.power',
        '_iblqc_ephysSpectralDensityAP.freqs',
    ]

    ephys_exists = all([da in ephys_data for da in dtypes_ephys])

    data = one.alyx.rest('datasets', 'list', subject=subj, date=date, collection='alf')
    trials_data = [str(Path(dat['name']).stem) for dat in data if '_ibl_trials' in dat['name']]
    trials_exists = len(trials_data) > 10
    wheel_data = [str(Path(dat['name']).stem) for dat in data if '_ibl_wheel' in dat['name']]
    wheel_exists = len(wheel_data) == 4
    dlc_data = [str(Path(dat['name']).stem) for dat in data if 'Camera.dlc' in dat['name']]
    me_data = [str(Path(dat['name']).stem) for dat in data if 'ROIMotionEnergy' in dat['name']]
    dlc_exists = (len(dlc_data) == 3 and len(me_data) == 6)
    passive_data_alf = [str(Path(dat['name']).stem) for dat in data if 'passive' in dat['name']]
    passive_alf_exists = len(passive_data_alf) == 4

    data = one.alyx.rest('datasets', 'list', subject=subj, date=date, collection='raw_passive_data')
    passive_data_raw = [str(Path(dat['name']).stem) for dat in data if 'RFMapStim' in dat['name']]
    passive_raw_exists = len(passive_data_raw) == 1

    passive_exists = all([passive_alf_exists, passive_raw_exists])

    histology = one.alyx.rest('sessions', 'list', subject=subj,
                              task_protocol='SWC_Histology_Serial2P_v0.0.1')
    histology_exists = len(histology) == 1

    planned_trajectory = one.alyx.rest('trajectories', 'list', subject=subj, date=date,
                                       probe=probe,
                                       provenance='Planned')
    planned_exists = len(planned_trajectory) == 1

    micro_trajectory = one.alyx.rest('trajectories', 'list', subject=subj, date=date, probe=probe,
                                     provenance='Micro-manipulator')
    micro_exists = len(micro_trajectory) == 1

    hist_trajectory = one.alyx.rest('trajectories', 'list', subject=subj, date=date, probe=probe,
                                    provenance='Histology track')

    hist_exists = len(hist_trajectory) == 1
    # if hist_exists and not histology_exists:
    #     histology_exists = True

    align_trajectory = one.alyx.rest('trajectories', 'list', subject=subj, date=date, probe=probe,
                                     provenance='Ephys aligned histology track')
    align_exists = len(align_trajectory) == 1

    insertion = one.alyx.rest('insertions', 'list', subject=subj, date=date, name=probe)
    if len(insertion) == 0:
        insertion_exists = False
        print(f'no insertion for {subj}, {date}, {probe}')
        align_resolved = False
    else:
        insertion_exists = True
        try:
            align_resolved = insertion[0].get('json').get('extended_qc').get('alignment_resolved',
                                                                             False)
        except Exception as err:
            print(f'no extended qc {subj}, {date}, {probe}')
            align_resolved = False

    if align_exists:
        traj = one.alyx.rest('trajectories', 'list', probe_insertion=align_trajectory[0]['probe_insertion'],
                             provenance='Ephys aligned histology track')
        users = [*traj[0]['json'].keys()]
        username = [us[20:] for us in users]
        user_note = str(username)
    else:
        user_note = ''

    status = {'Subject': subj, 'Date': date, 'Probe': probe, 'ks2': ks2_exists,
              'raw_ephys_qc': ephys_exists, 'trials': trials_exists, 'wheel': wheel_exists,
              'dlc': dlc_exists, 'passive': passive_exists, 'histology': histology_exists,
              'insertion': insertion_exists, 'planned': planned_exists, 'micro': micro_exists,
              'tracing': hist_exists, 'aligned': align_exists, 'resolved': align_resolved,
              'user_note': user_note}

    return status
