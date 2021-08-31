"""
Update G-sheet rep.site status
Author: Mayo, Gaelle
(turn into function for ease of call - GC)
"""


def update_rep_site():

    from googleapiclient.discovery import build
    from httplib2 import Http
    from oauth2client import client, file, tools
    from os.path import join
    import pandas as pd
    import numpy as np
    from oneibl.one import ONE
    from repeated_site_data_status import get_repeated_site_status
    # import brainbox.behavior.training as training
    from reproducible_ephys_functions import query, STR_QUERY, exclude_recordings, data_path

    one = ONE()

    # Define paths to authentication details
    '''
    # Mayo's
    credentials_file_path = '/iblmayo/googleAPI/credentials/IBL/credentials.json'
    clientsecret_file_path = '/iblmayo/googleAPI/credentials/IBL/client_secret_IBL.json'
    '''
    # Gaelle's
    credentials_file_path = '/Users/gaelle/Documents/Work/Google_IBL/credentials.json'
    clientsecret_file_path = '/Users/gaelle/Documents/Work/Google_IBL/client_secret.json'

    SCOPE = ['https://www.googleapis.com/auth/spreadsheets']
    # See if credentials exist
    store = file.Storage(credentials_file_path)
    credentials = store.get()

    # If not get new credentials
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(clientsecret_file_path, SCOPE)
        credentials = tools.run_flow(flow, store)

    drive_service = build('drive', 'v3', http=credentials.authorize(Http()))
    sheets = build('sheets', 'v4', http=credentials.authorize(Http()))

    read_spreadsheetID = '1uJqGuoPBzn1GlAmgcxilSR2gEUjHFYLyH3SKAMV0QwI'
    read_spreadsheetRange = 'repeated site'
    rows = sheets.spreadsheets().values().get(spreadsheetId=read_spreadsheetID,
                                              range=read_spreadsheetRange).execute()
    data = pd.DataFrame(rows.get('values'))
    data = data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)

    # Get data from alignment sheet
    read_spreadsheetID_align = '1nidCu7MjLrjaA8NHWYnJavLzQBjILZhCkYt0OdCUTxg'
    read_spreadsheetRange_align = 'NEW_2'
    rows_align = sheets.spreadsheets().values().get(spreadsheetId=read_spreadsheetID_align,
                                                    range=read_spreadsheetRange_align).execute()

    data_sheet_align = pd.DataFrame(rows_align.get('values'))
    data_sheet_align = data_sheet_align.rename(columns=data_sheet_align.iloc[0]).drop(data_sheet_align.index[0]).reset_index(drop=True)


    # Clean up the data a bit
    subjects = data['Mouse ID'].values[1:]
    subjects[subjects == 'DY_010 (C)'] = 'DY_010'
    dates = data['Date'].values[1:]
    probes = data['Probe ID'].values[1:]
    probes = np.array(['probe0' + pr if 'probe0' not in pr else pr for pr in probes])

    # For NYU and witten change from gsheet as they messed up!
    probes[subjects == 'NYU-21'] = 'probe00'
    # probes[subjects == 'ibl_witten_13'] = 'probe00'  # GC note: comment out as probe01 seems correct
    probes[subjects == 'NYU-11'] = 'Probe00'

    gsheet_list = [subj + '*' + date + '*' + probe for subj, date, probe in
                   zip(subjects, dates, probes)]


    trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                 x=-2243, y=-2000,  # repeated site coordinate
                                 project='ibl_neuropixel_brainwide_01',
                                 django='probe_insertion__session__qc__lt,50')
    alyx_list = [traj['session']['subject'] + '*' + traj['session']['start_time'][0:10] + '*' +
                 traj['probe_name'] for traj in trajectories]

    extra_list = np.setdiff1d(np.array(alyx_list), np.array(gsheet_list))
    for ex in extra_list:
        info = ex.split('*')
        subjects = np.append(subjects, info[0])
        dates = np.append(dates, info[1])
        probes = np.append(probes, info[2])

    probes[subjects == 'NYU-11'] = 'probe00'

    df = pd.DataFrame(columns={'Subject', 'Date', 'Probe', 'ks2', 'raw_ephys_qc', 'trials', 'wheel',
                               'dlc', 'passive', 'histology', 'insertion', 'planned', 'micro',
                               'tracing', 'aligned', 'resolved', 'user_note', 'origin_lab', 'assign_lab'})

    # get insertions used in analysis
    q = query()
    q_ins_id = [item['probe_insertion'] for item in q]
    del q

    # get insertions potentially good
    q = one.alyx.rest('trajectories', 'list', django=STR_QUERY)
    q_ins_potential = [item['probe_insertion'] for item in q]
    del q

    # get insertions that are potentially good but do not match traj coord
    q = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                 x=-2243, y=-2000, theta=15,
                                 django=STR_QUERY)
    # TODO should be equivalent to query(resolved=False, min_regions=0)
    q_ins_coordcorrect = [item['probe_insertion'] for item in q]
    del q

    # get insertions that are passing L1 QC
    q = one.alyx.rest('trajectories', 'list', provenance='Planned',
                      django='probe_insertion__session__project__name__'
                             'icontains,ibl_neuropixel_brainwide_01,'
                             'probe_insertion__session__qc__lt,50,'  # TODO add insertion not CRITICAL
                             'probe_insertion__session__extended_qc__behavior,1,'
                             'probe_insertion__json__extended_qc__tracing_exists,True,'
                             '~probe_insertion__session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_response_feedback_delays__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_response_stimFreeze_delays__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_n_trial_events__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_reward_volumes__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_reward_volume_set__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
                             '~probe_insertion__session__extended_qc___task_audio_pre_trial__lt,0.9')
    q_ins_passl1 = [item['probe_insertion'] for item in q]

    # Get reason for exclusion by additional criteria
    metrics = pd.read_csv(join(data_path(), 'metrics_region.csv'))
    _, excl_rec = exclude_recordings(metrics, return_excluded=True)

    for subj, date, probe in zip(subjects, dates, probes):
        print(f'====== {subj}, {date}, {probe} ======')  # todo remove, for debugging
        status = get_repeated_site_status(subj, date, probe, one=one)

        # Check if insertion is used in analysis as per query
        insertion = one.alyx.rest('insertions', 'list', subject=subj, date=date, name=probe)
        if len(insertion) == 0:
            is_used_analysis = False
            is_potential = False
            ins_id = "NaN"
        else:
            ins = insertion[0]
            ins_id = ins['id']
            if ins_id in q_ins_id:
                is_used_analysis = True
            else:
                is_used_analysis = False

            if ins_id in q_ins_potential:
                is_potential = True
            else:
                is_potential = False

            if ins_id in q_ins_coordcorrect:
                is_coordcorrect = True
            else:
                is_coordcorrect = False

            if ins_id in q_ins_passl1:
                is_passl1 = True
            else:
                is_passl1 = False

        status['is_used_analysis'] = is_used_analysis
        status['is_potential'] = is_potential
        status['is_coordcorrect'] = is_coordcorrect
        status['is_passl1'] = is_passl1
        status['ins_id'] = ins_id

        # Get excluded reasons
        status['high_noise'] = excl_rec.loc[excl_rec['subject'] == subj, 'high_noise']
        status['low_yield'] = excl_rec.loc[excl_rec['subject'] == subj, 'low_yield']
        status['missed_target'] = excl_rec.loc[excl_rec['subject'] == subj, 'missed_target']

        # Use ins_id to find who is assigned to do alignment
        if data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id].empty or \
           data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id].empty:
            status['origin_lab'] = 'NOT FOUND'
            status['assign_lab'] = 'NOT FOUND'
            print(f'Insertion {ins_id} NOT FOUND')
        else:
            status['origin_lab'] = data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id, 'origin_lab'].iloc[0]
            status['assign_lab'] = data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id, 'assign_lab'].iloc[0]

        # User note - requires trajectory
        trajs = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                              subject=subj, date=date, probe=probe)
        status['usernote'] = ''
        if len(trajs) > 0:
            traj = trajs[0]
            json_field = traj['json']
            if json_field is not None:
                key_json = list(json_field.keys())

                if len(key_json) > 0:
                    str_note_all = ''
                    for key_i in key_json:
                        str_note = json_field[key_i][-1]
                        if type(str_note) == str:
                            str_note_all = str_note_all + key_i + ': ' + str_note + ' ; '
                            # print(str_note_all)
                            status['usernote'] = str_note_all

        # append to DF
        df = df.append(status, ignore_index=True)



    df = df.reindex(columns=['ins_id', 'Subject', 'Date', 'Probe',
                             'is_potential', 'is_coordcorrect',  'is_used_analysis', 'is_passl1',
                             'ks2', 'raw_ephys_qc', 'trials', 'wheel',
                             'dlc', 'passive', 'histology', 'insertion', 'planned', 'micro',
                             'tracing', 'aligned', 'resolved', 'high_noise', 'low_yield',
                             'missed_target', 'user_note', 'origin_lab', 'assign_lab',
                             'usernote'])

    df = df.sort_values(by=['Subject', 'Date'], ascending=True)

    write_spreadsheetID = '1pRLFvyVgmIJfKSX4GqmmAFuNRTpU1NvMlYOijq7iNIA'
    write_spreadsheetRange = 'Sheet1'

    write_data = sheets.spreadsheets().\
        values().update(spreadsheetId=write_spreadsheetID, valueInputOption='RAW',
                        range=write_spreadsheetRange,
                        body=dict(majorDimension='ROWS',
                                  values=df.T.reset_index().T.values.tolist())).execute()
    print('Sheet successfully Updated')

    my_range = {
        'sheetId': 0,
        'startRowIndex': 1,
        'endRowIndex': len(df) + 1,
        'startColumnIndex': 8,
        'endColumnIndex': 18  # len(df.columns),
    }
    requests = [{
        'setDataValidation': {
              "range": my_range,
              "rule": {
                "condition": {
                  'type': 'BOOLEAN',
                },
                "inputMessage": 'la',
                "strict": True,
                "showCustomUi": False
              }
        }
    }]
    body = {
        'requests': requests
    }
    response = sheets.spreadsheets() \
        .batchUpdate(spreadsheetId=write_spreadsheetID, body=body).execute()

    print('{0} cells updated.'.format(len(response.get('replies'))))
