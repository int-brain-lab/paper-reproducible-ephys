from one.webclient import http_download_file
from reproducible_ephys_functions import save_data_path


def download_meshes(atlas_id):
    url = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'
    save_dir = save_data_path(figure='figure1').joinpath('meshes')
    save_dir.mkdir(exist_ok=True, parents=True)
    mesh_url = url + str(atlas_id) + '.obj'
    http_download_file(mesh_url, target_dir=save_dir)


