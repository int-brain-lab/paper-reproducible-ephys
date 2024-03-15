"""
Report on the reason why repeated sites were abandoned from analysis.
"""
# Author: Gaelle C.

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import client, file, tools
import pandas as pd
import numpy as np
from oneibl.one import ONE
from reproducible_ephys_functions import STR_QUERY

one = ONE()

# Define paths to authentication details

# Gaelle's credentials
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

read_spreadsheetID = '1pRLFvyVgmIJfKSX4GqmmAFuNRTpU1NvMlYOijq7iNIA'
read_spreadsheetRange = 'Sheet1'
rows = sheets.spreadsheets().values().get(spreadsheetId=read_spreadsheetID,
                                          range=read_spreadsheetRange).execute()
data = pd.DataFrame(rows.get('values'))
data = data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)

# Get insertions ID from datasheet
ins_data = np.unique(data['ins_id'])   # Note: trailing rows will appear as ''
ins_data = np.delete(ins_data, np.where(ins_data == ''))

print(f'Total unique ins ID on sheet: {len(ins_data)}')

# get insertions potentially good
q = one.alyx.rest('trajectories', 'list', django=STR_QUERY)
q_ins_potential = [item['probe_insertion'] for item in q]
del q

'''
DECOMPOSE STR QUERY into many sub-queries
STR_QUERY = 'probe_insertion__session__projects__name__icontains,ibl_neuropixel_brainwide_01,' \
            'probe_insertion__session__qc__lt,50,' \
            '~probe_insertion__json__qc,CRITICAL,' \
            'probe_insertion__session__n_trials__gte,400'
'''

# get insertions with project code
q = one.alyx.rest('insertions', 'list', django='session__projects__name__icontains,ibl_neuropixel_brainwide_01')
q_ins_project = [item['id'] for item in q]
del q

# get insertions for which ins>qc is not marked as CRITICAL
q = one.alyx.rest('insertions', 'list', django='~json__qc,CRITICAL')
q_ins_inscrit = [item['id'] for item in q]
del q

# get insertions for which sess>qc is not marked as CRITICAL
q = one.alyx.rest('insertions', 'list', django='session__qc__lt,50')
q_ins_sesscrit = [item['id'] for item in q]
del q

# get insertions for which ntrial is >=400
q = one.alyx.rest('insertions', 'list', django='session__n_trials__gte,400')
q_ins_ntriallow = [item['id'] for item in q]
del q

# Init vect
v_ins_potential = []

v_ins_project = []
v_ins_inscrit = []
v_ins_sesscrit = []
v_ins_ntriallow = []

# Loop over insertion and place into categories (can be overlapping)
for ins in ins_data:
    if ins in q_ins_potential:
        v_ins_potential.append(ins)
    # Decomposition of failures:
    if ins not in q_ins_project:
        v_ins_project.append(ins)
    if ins not in q_ins_inscrit:
        v_ins_inscrit.append(ins)
    if ins not in q_ins_sesscrit:
        v_ins_sesscrit.append(ins)
    if ins not in q_ins_ntriallow:
        v_ins_ntriallow.append(ins)

# Todo check if any overlap
print(f'Ins potential : {len(v_ins_potential)}')
print(f'Ins with wrong project code : {len(v_ins_project)}')
print(f'Ins with ins>qc marked as CRITICAL : {len(v_ins_inscrit)}')
print(f'Ins with sess>qc marked as CRITICAL : {len(v_ins_sesscrit)}')
print(f'Ins with low trial n : {len(v_ins_ntriallow)}')

