"""
Update G-sheet rep.site status (Version 2)
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
    from one.api import ONE
    from repeated_site_data_status import get_repeated_site_status
    # import brainbox.behavior.training as training
    from reproducible_ephys_functions import query, STR_QUERY, exclude_recordings, data_path

    one = ONE()

    # Define paths to authentication details
    # Mayo's
    # credentials_file_path = r'C:\Users\Mayo\iblenv\iblmayo\googleAPI\credentials\IBL\credentials.json'
    # clientsecret_file_path = r'C:\Users\Mayo\iblenv\iblmayo\googleAPI\credentials\IBL\client_secret_IBL.json'

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

    # Get repeated sites
    trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                 x=-2243, y=-2000,  # repeated site coordinate
                                 django=STR_QUERY)

    subjects = [traj['session']['subject'] for traj in trajectories]
    dates = [traj['session']['start_time'][0:10] for traj in trajectories]
    probes = [traj['probe_name'] for traj in trajectories]

    # Get reason for exclusion by additional criteria
    metrics = pd.read_csv(join(data_path(), 'metrics_region.csv'))
    _, excl_rec = exclude_recordings(metrics, return_excluded=True)

    # Init dataframe
    df = pd.DataFrame(columns={'Subject', 'Date', 'Probe', 'ks2', 'raw_ephys_qc', 'trials', 'wheel',
                               'dlc', 'passive', 'histology', 'insertion', 'planned', 'micro',
                               'tracing', 'aligned', 'resolved', 'user_note', 'origin_lab', 'assign_lab'})


    for subj, date, probe in zip(subjects, dates, probes):
        print(f'====== {subj}, {date}, {probe} ======')  # todo remove, for debugging
        status = get_repeated_site_status(subj, date, probe, one=one)

        # Check if insertion is used in analysis as per query
        insertion = one.alyx.rest('insertions', 'list', subject=subj, date=date, name=probe)
        ins = insertion[0]
        ins_id = ins['id']

        status['ins_id'] = ins_id

        # Get excluded reasons
        atest = excl_rec.loc[excl_rec['subject'] == subj, 'high_noise'].values
        if bool(atest) is True:
            status['high_noise'] = True
        else:
            status['high_noise'] = False
        del atest

        atest = excl_rec.loc[excl_rec['subject'] == subj, 'low_yield'].values
        if bool(atest) is True:
            status['low_yield'] = True
        else:
            status['low_yield'] = False
        del atest

        atest = excl_rec.loc[excl_rec['subject'] == subj, 'missed_target'].values
        if bool(atest) is True:
            status['missed_target'] = True
        else:
            status['missed_target'] = False
        del atest

        # Use ins to find who is assigned to do alignment
        status['origin_lab'] = ins['session_info']['lab']
        if 'todo_alignment' not in ins['json'].keys():
            status['assign_lab'] = 'NOT FOUND'
        else:
            status['assign_lab'] = ins['json']['todo_alignment']

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
        # df = df.append(status, ignore_index=True)
        df_status = pd.DataFrame(status, index=[0])
        df = pd.concat((df, df_status), axis=0)


    df = df.reindex(columns=['ins_id', 'Subject', 'Date', 'Probe',
                             'ks2', 'raw_ephys_qc', 'trials', 'wheel',
                             'dlc', 'passive', 'histology', 'insertion', 'planned', 'micro',
                             'tracing', 'aligned', 'resolved', 'high_noise', 'low_yield',
                             'missed_target', 'user_note', 'origin_lab', 'assign_lab',
                             'usernote'])

    df = df.sort_values(by=['Subject', 'Date'], ascending=True)

    write_spreadsheetID = '1pRLFvyVgmIJfKSX4GqmmAFuNRTpU1NvMlYOijq7iNIA'
    write_spreadsheetRange = 'Copy of Sheet1'

    write_data = sheets.spreadsheets().\
        values().update(spreadsheetId=write_spreadsheetID, valueInputOption='RAW',
                        range=write_spreadsheetRange,
                        body=dict(majorDimension='ROWS',
                                  values=df.T.reset_index().T.values.tolist())).execute()
    print('Sheet successfully Updated')
    #
    # my_range = {
    #     'sheetId': 0,
    #     'startRowIndex': 1,
    #     'endRowIndex': len(df) + 1,
    #     'startColumnIndex': 8,
    #     'endColumnIndex': 18  # len(df.columns),
    # }
    # requests = [{
    #     'setDataValidation': {
    #           "range": my_range,
    #           "rule": {
    #             "condition": {
    #               'type': 'BOOLEAN',
    #             },
    #             "inputMessage": 'la',
    #             "strict": True,
    #             "showCustomUi": False
    #           }
    #     }
    # }]
    # body = {
    #     'requests': requests
    # }
    # response = sheets.spreadsheets() \
    #     .batchUpdate(spreadsheetId=write_spreadsheetID, body=body).execute()
    #
    # print('{0} cells updated.'.format(len(response.get('replies'))))
