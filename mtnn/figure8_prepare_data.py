from utils import get_mtnn_eids, get_traj, featurize
from reproducible_ephys_functions import save_data_path, save_dataset_info
from one.api import ONE
from ibllib.atlas import AllenAtlas
import tqdm
import numpy as np

one = ONE()
one.record_loaded = True
ba = AllenAtlas()
mtnn_eids = get_mtnn_eids()
traj = get_traj(mtnn_eids)


feature_list = []
output_list = []
cluster_number_list = []
trial_number_list = []
session_list = []
session_count = {'mainenlab': 0, 'churchlandlab': 0,
                 ('hoferlab', 'mrsicflogellab'): 0,
                 'danlab': 0, 'angelakilab': 0}

for i in range(len(traj)):
    feature, output, cluster_numbers, trial_numbers = featurize(i, traj[i], one, session_count, brain_atlas=ba)
    feature_list.append(feature)
    output_list.append(output)
    cluster_number_list.append(cluster_numbers)
    session_list.append(traj[i])
    trial_number_list.append(trial_numbers)

save_dataset_info(one, figure='figure8')

save_path = save_data_path(figure='figure8').joinpath('original_data')
save_path.mkdir(exist_ok=True, parents=True)

for i in range(len(feature_list)):
    print(session_list[i]['session']['id'])
    print(feature_list[i].shape)
    np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_feature.npy'), feature_list[i])


for i in range(len(output_list)):
    print(session_list[i]['session']['id'])
    print(output_list[i].shape)
    np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_output.npy'), output_list[i])


for i in range(len(cluster_number_list)):
    print(session_list[i]['session']['id'])
    print(cluster_number_list[i].shape)
    np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_clusters.npy'), cluster_number_list[i])

for i in range(len(session_list)):
    print(session_list[i]['session']['id'])
    np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_session_info.npy'), session_list[i])

for i in range(len(trial_number_list)):
    print(session_list[i]['session']['id'])
    print(trial_number_list[i].shape)
    np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_trials.npy'), trial_number_list[i])
