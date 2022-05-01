from figure4.figure4_load_data import load_data, load_dataframe
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs, save_figure_path, figure_style

df = load_dataframe()
df_filt = filter_recordings(df)

subject = 'CSHL052'
region = 'CA1'

print(df_filt.loc[(df['region'] == region) & (df['subject'] == subject)].institute)

df_filt = df_filt[df_filt['include'] == 1].reset_index()

print(df_filt.loc[(df['region'] == region) & (df['subject'] == subject)].institute)
