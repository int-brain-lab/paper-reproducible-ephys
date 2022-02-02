FIGURE_5 = {'channels': ['brainLocationIds_ccf_2017', 'localCoordinates', 'mlapdv', 'rawInd'],
            'clusters': ['amps', 'channels', 'metrics', 'peak2trough'],
            'spikes': ['amps', 'clusters', 'depths', 'times'],
            'trials': ['choice', 'contrastLeft', 'contrastRight', 'feedbackType', 'feedback_times',
                       'firstMovement_times', 'stimOn_times']}


def load_figure_5_data(insertions, one=None, ba=None):
    one = one or ONE()
    ba = ba or AllenAtlas()



    #
