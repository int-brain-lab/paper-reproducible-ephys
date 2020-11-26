from features_2D import plot_2D_features
from distance_from_repeated_site import sort_repeated_site_by_distance
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import client, file, tools
import pandas as pd
from oneibl.one import ONE
from ibllib.atlas import atlas

# Define paths to authentication details
credentials_file_path = 'C:/Users/Mayo/iblenv/iblmayo/googleAPI/credentials/IBL/credentials.json'
clientsecret_file_path = 'C:/Users/Mayo/iblenv/iblmayo/googleAPI/credentials/IBL/client_secret_IBL.json'

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

# Get data from repeated site status googlesheet
read_spreadsheetID = '1pRLFvyVgmIJfKSX4GqmmAFuNRTpU1NvMlYOijq7iNIA'
read_spreadsheetRange = 'Sheet1'
rows = sheets.spreadsheets().values().get(spreadsheetId=read_spreadsheetID,
                                          range=read_spreadsheetRange).execute()
data = pd.DataFrame(rows.get('values'))
data = data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)

# Find the sessions that have raw_ephys data and tracing
idx_ephys = data['raw_ephys'] == 'TRUE'
idx_ephys = data['ks2'] == 'TRUE'
idx_tracing = data['tracing'] == 'TRUE'
subjects = data['Subject'][idx_ephys & idx_tracing].values
dates = data['Date'][idx_ephys & idx_tracing].values
probes = data['Probe'][idx_ephys & idx_tracing].values


one = ONE()
brain_atlas = atlas.AllenAtlas(25)

# Sort insertions by distance from mean trajectory across all sites
_, idx_sort = sort_repeated_site_by_distance(subjects, dates, probes, reference='mean', one=one,
                                             brain_atlas=brain_atlas)
#idx_sort = idx_sort[:19]
# Plot the features
# TODO pass in the type of feature that you want to display
plot_2D_features(subjects[idx_sort], dates[idx_sort], probes[idx_sort], one=one,
                 brain_atlas=brain_atlas, plot_type='amp_line')
