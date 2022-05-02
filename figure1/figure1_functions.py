from one.webclient import http_download_file
from reproducible_ephys_functions import save_data_path
from ibllib.atlas.regions import BrainRegions


def download_meshes(atlas_id):
    save_dir = save_data_path(figure='figure1').joinpath('meshes')
    save_dir.mkdir(exist_ok=True, parents=True)
    if save_dir.joinpath(f'{atlas_id}.obj').exists():
        return
    url = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'
    save_dir = save_data_path(figure='figure1').joinpath('meshes')
    save_dir.mkdir(exist_ok=True, parents=True)
    mesh_url = url + str(atlas_id) + '.obj'
    http_download_file(mesh_url, target_dir=save_dir)


def add_mesh(fig, obj_file, color=(1., 1., 1.), opacity=0.4):
    """
    Adds a mesh object from an *.obj file to the mayavi figure
    :param fig: mayavi figure
    :param obj_file: full path to a local *.obj file
    :param color: rgb tuple of floats between 0 and 1
    :param opacity: float between 0 and 1
    :return: vtk actor
    """

    import vtk

    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(obj_file))
    reader.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    fig.scene.add_actor(actor)
    fig.scene.render()
    return mapper, actor


def add_br_meshes(fig, brain_areas=None, opacity=0.6, br=None):

    br = br or BrainRegions()

    brain_areas = brain_areas or ['root', 'VISa', 'CA1', 'DG', 'LP', 'PO']
    for area in brain_areas:
        atlas_id = br.acronym2id(area)[0]
        mesh_path = save_data_path(figure='figure1').joinpath('meshes', f'{atlas_id}.obj')
        _, idx = br.id2index(atlas_id)
        color = br.rgb[idx[0][0], :] / 255
        add_mesh(fig, mesh_path, color, opacity=opacity)
