'''
Fig1 subpanel
Raster plot + histology brain regions
'''
# Author: Gaelle

import matplotlib.pyplot as plt
from ibllib.ephys import neuropixel
from one.api import ONE
from reproducible_ephys_functions import query
from pathlib import Path
import random
import brainbox.plot as bbplot
# from brainbox.io.one import load_spike_sorting_with_channel
from brainbox.io.one import load_spike_sorting_fast
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from brainbox.ephys_plots import plot_brain_regions


one = ONE()
h = neuropixel.trace_header()
ba = AllenAtlas()
br = BrainRegions()


n_plt_raw = 1
vect_plt = [[0.5] * 2, [6] * n_plt_raw]
flat_vect_plt = [item for sublist in vect_plt for item in sublist]
n_plt = len(flat_vect_plt)

OVERWRITE = False

fig_path = Path.home().joinpath('Desktop/Overview_plot_BWM/Repeatedsite/Raster_Fig1_V2')
fig_path.mkdir(parents=True, exist_ok=True)

# Get pid used in Repeated site analysis
q = query()
pids_rs = [item['probe_insertion'] for item in q]

# Select pids to plot
pids = pids_rs
len_pids = len(pids)

random.shuffle(pids)

for i_pid, pid in enumerate(pids):

    # GET RECORDING INFO
    eid, pname = one.pid2eid(pid)
    sess_path = one.eid2path(eid)
    prob_path = str(Path(sess_path).joinpath(pname))
    path_name = prob_path[33:]  # Note: hardcoded 33 because GC path is `/Users/gaelle/Downloads/FlatIron/`
    path_name_und = path_name.replace('/', '_')
    fig_name = f"{pid}__{path_name_und}__destripe"
    figname = fig_path.joinpath(fig_name)
    if figname.exists() and not OVERWRITE:
        continue

    print(f'------------ {i_pid}/{len_pids} ------------')

    # GET SPIKE SORTING DATA
    fig, axs = plt.subplots(nrows=1, ncols=n_plt, figsize=(14, 5), gridspec_kw={'width_ratios': flat_vect_plt})
    fig.suptitle(path_name)
    spikes, clusters, channels = load_spike_sorting_fast(eid=eid, one=one, probe=pname,
                                                         dataset_types=['spikes.amps', 'spikes.depths'])

    # spikes, clusters, channels = load_spike_sorting_with_channels(eid=eid, one=one, probe=pname,
    #                                                              dataset_types=['spikes.amps', 'spikes.depths'],
    #                                                              brain_atlas=ba)

    # unnest
    spikes, clusters, channels = spikes[pname], clusters[pname], channels[pname]
    # PLOT
    # HISTOLOGY
    if 'atlas_id' in channels.keys():
        mapped_ids = br.remap(channels['atlas_id'], source_map='Allen', target_map='Cosmos')
        # Plot brain regions
        plot_brain_regions(channel_ids=channels['atlas_id'], channel_depths=None, brain_regions=None, display=True,
                           ax=axs[0])
        plot_brain_regions(channel_ids=mapped_ids, channel_depths=None, brain_regions=None, display=True, ax=axs[1])

    # RASTER
    bbplot.driftmap(spikes['times'],
                    spikes['depths'],
                    ax=axs[2], plot_style='bincount')

    # Save plot
    fig.tight_layout()
    fname = fig_path.joinpath(f'{pid}_RasterHist.pdf')
    plt.savefig(fname=fname)
    plt.close(fig)
