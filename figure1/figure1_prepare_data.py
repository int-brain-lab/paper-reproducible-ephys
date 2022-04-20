from figure1.figure1_functions import download_meshes
from ibllib.atlas.regions import BrainRegions

br = BrainRegions()
brain_regions = ['VISa', 'CA1', 'DG', 'LP', 'PO']


for region in brain_regions:
    atlas_id = br.acronym2id(region)[0]
    download_meshes(atlas_id)
    print(atlas_id)



