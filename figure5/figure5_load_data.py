FIGURE_5 = {'channels': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv', 'rawInd'],
            'clusters': ['amps', 'channels', 'metrics', 'peak2trough'],
            'spikes': ['amps', 'clusters', 'depths', 'times'],
            'trials': ['choice', 'contrastLeft', 'contrastRight', 'feedbackType', 'feedback_times',
                       'firstMovement_times', 'stimOn_times']}

import pandas as pd
from reproducible_ephys_functions import save_data_path


def load_dataframe(exists_only=False):
    df_path = save_data_path(figure='figure5').joinpath('figure5_dataframe.csv')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return None

