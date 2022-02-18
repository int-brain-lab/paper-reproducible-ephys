from reproducible_ephys_functions import get_insertions, combine_regions, BRAIN_REGIONS
from one.api import ONE, One
one = ONE()
one_local = One()
ins = get_insertions(level=2, one=one)

# Load in spike sorting data

sl = SpikeSortingLoader(eid=eid, pname=probe, one=one_local, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
clusters = sl.merge_clusters(spikes, clusters, channels)
clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
# Find clusters that are in the repeated site brain regions and that have been labelled as good
cluster_idx = np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1))[0]


# basically need to loop through and get what is

