# DEMO 1: add insertion probes data from IBL database (DataJoints)
from iblviewer import atlas_controller
import oneibl.one
import numpy as np


def get_bwm_ins_alyx(one):
    """
    Return insertions that match criteria :
    - project code
    - session QC not critical (TODO may need to add probe insertion QC)
    - at least 1 alignment
    - behavior pass
    :param one: "one" connection handler
    :return:
    ins: dict containing the full details on insertion as per the alyx rest query
    ins_id: list of insertions eids
    sess_id: list of (unique) sessions eids
    """
    STR_QUERY = 'probe_insertion__session__projects__name__icontains,ibl_neuropixel_brainwide_01,' \
                'probe_insertion__session__qc__lt,50,' \
                '~probe_insertion__json__qc,CRITICAL,' \
                'probe_insertion__session__n_trials__gte,400,' \
                'probe_insertion__json__extended_qc__alignment_resolved,True'

    traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                        x=-2243, y=-2000, theta=15, django=STR_QUERY)

    ins_id = [item['probe_insertion'] for item in traj]
    sess_id = [item['session']['id'] for item in traj]
    # Here's what's in 'json':
    # dict_keys(['qc', 'n_units', 'xyz_picks', 'extended_qc', 'drift_rms_um', 'firing_rate_max', 'n_units_qc_pass',
    # 'amplitude_max_uV', 'firing_rate_median', 'amplitude_median_uV', 'whitening_matrix_conditioning'])
    xyz_picks = {}
    for item in ins_id:
        ins = one.alyx.rest('insertions', 'list', id=item)
        xyz_picks[item] = np.array(ins[0]['json'].get('xyz_picks', []))
    sess_id = np.unique(sess_id)
    return xyz_picks


def get_picks_mean_vectors(xyz_picks, extent=3):
    """
    Get a mean vector from picks coordinates
    :param xyz_picks: Dictionary xyz picks, the key being the identifier for that data set
    :param extent: Number of points to take from start and end for mean computation of end points
    :return: 3D numpy array and a list of ids
    """
    vectors = []
    ids = []
    # Mean between first and last three picks
    for ins_id in xyz_picks:
        raw_picks = xyz_picks[ins_id]
        end_pt = np.mean(raw_picks[-extent:], axis=0)
        start_pt = np.mean(raw_picks[:extent], axis=0)
        vectors.append([start_pt, end_pt])
        ids.append(ins_id)
    return np.array(vectors), ids


def add_insertion_probes(controller, one_connection, reduced=True, line_width=5):
    """
    Add insertion probe vectors
    :param controller: The IBLViewer controller
    :param one_connection: The "one" connection to IBL server
    :param reduced: Whether insertion probes should be reduced to simple lines
    :param with_labels: Whether labels should be added to the lines
    """
    vectors = get_bwm_ins_alyx(one_connection)
    if reduced:
        vectors, ids = get_picks_mean_vectors(vectors)
        lines = controller.view.add_segments(vectors, line_width=line_width)
    else:
        lines = controller.view.add_lines(vectors, line_width=line_width)
    actors = [lines]

    controller.plot.add(actors)
    return lines


if __name__ == '__main__':

    one_connection = oneibl.one.ONE(base_url="https://alyx.internationalbrainlab.org")
    controller = atlas_controller.AtlasController()
    controller.initialize(resolution=25, mapping='Allen', embed_ui=True, jupyter=False)

    add_insertion_probes(controller, one_connection, reduced=True)
    controller.render()
