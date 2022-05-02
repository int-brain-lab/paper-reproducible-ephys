from figure1.figure1_functions import download_meshes
from figure1.figure1_load_data import mesh_regions, example_pid, suppl_pids
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE


def prepare_data(one, ba=None):
    ba = ba or AllenAtlas()

    pids = [example_pid] + [ids for vals in suppl_pids.values() for ids in vals]

    for pid in pids:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        _ = sl.load_spike_sorting()

    for region in mesh_regions:
        atlas_id = ba.regions.acronym2id(region)[0]
        download_meshes(atlas_id)


if __name__ == '__main__':
    one = ONE()
    prepare_data(one=one)
