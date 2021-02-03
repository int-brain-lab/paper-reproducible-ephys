"""
Update G-sheet rep.site status
Author: Mayo, Gaelle
(turn into function for ease of call - GC)
"""


def update_rep_site():

    from googleapiclient.discovery import build
    from httplib2 import Http
    from oauth2client import client, file, tools
    import pandas as pd
    import numpy as np
    from oneibl.one import ONE
    from repeated_site_data_status import get_repeated_site_status
    import brainbox.behavior.training as training

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
    probes[subjects == 'ibl_witten_13'] = 'probe00'
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

    df = pd.DataFrame(columns={'Subject', 'Date', 'Probe', 'ks2', 'raw_ephys', 'trials', 'wheel',
                               'dlc', 'passive', 'histology', 'insertion', 'planned', 'micro',
                               'tracing', 'aligned', 'resolved', 'user_note', 'origin_lab', 'assign_lab'})

    for subj, date, probe in zip(subjects, dates, probes):
        status = get_repeated_site_status(subj, date, probe, one=one)

        # Check if insertion is critical, criteria (OR):
        # - session critical
        # - insertion critical
        # - behavior fail
        # - impossible to trace
        insertion = one.alyx.rest('insertions', 'list', subject=subj, date=date, name=probe)
        if len(insertion) == 0:
            is_critical = False
            ins_id = 'NaN'
        else:
            ins = insertion[0]
            ins_id = ins['id']
            eid = ins['session_info']['id']
            sess_crit = one.alyx.rest('sessions', 'list', id=eid,
                                      django='qc,50')
            behav_crit = one.alyx.rest('sessions', 'list', id=eid,
                                       django='extended_qc__behavior,0')
            ins_crit = one.alyx.rest('insertions', 'list', id=ins['id'],
                                     django='json__qc__icontains,CRITICAL')
            trac_crit = one.alyx.rest('insertions', 'list', id=ins['id'],
                                      django='json__extended_qc__tracing_exists,False')

            if len(sess_crit) > 0 or len(behav_crit) > 0 or len(ins_crit) > 0 or len(trac_crit) > 0:
                is_critical = 'FAIL'
                # Check if only behavior status fails
                if len(behav_crit) > 0 and \
                        len(sess_crit) == 0 and \
                        len(ins_crit) == 0 and \
                        len(trac_crit) == 0:
                    # Compute behavior N trials and perf
                    trials_all = one.load_object(eid, 'trials')
                    trials = dict()
                    trials['temp_key'] = trials_all
                    perf_easy, n_trials, _, _, _ = training.compute_bias_info(trials, trials_all)
                    if perf_easy > 0.88 and n_trials >= 400:
                        is_critical = 'BORDERLINE'
            else:
                is_critical = 'PASS'
        status['is_critical'] = is_critical
        status['ins_id'] = ins_id

        # Use ins_id to find who is assigned to do alignment
        if data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id].empty or \
           data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id].empty:
            status['origin_lab'] = 'NOT FOUND'
            status['assign_lab'] = 'NOT FOUND'
            print(f'Insertion {ins_id} NOT FOUND')
        else:
            status['origin_lab'] = data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id, 'origin_lab'].iloc[0]
            status['assign_lab'] = data_sheet_align.loc[data_sheet_align['ins_id'] == ins_id, 'assign_lab'].iloc[0]

        df = df.append(status, ignore_index=True)

    df = df.reindex(columns=['ins_id', 'Subject', 'Date', 'Probe', 'is_critical', 'ks2', 'raw_ephys', 'trials', 'wheel',
                             'dlc', 'passive', 'histology', 'insertion', 'planned', 'micro',
                             'tracing', 'aligned', 'resolved', 'user_note', 'origin_lab', 'assign_lab'])

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
        'startColumnIndex': 4,
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
