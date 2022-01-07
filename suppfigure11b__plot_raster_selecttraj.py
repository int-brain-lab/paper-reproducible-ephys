'''
Supplement Fig1-1b
Raster plot + histology brain regions for selected traj
'''
# Author: Gaelle

import matplotlib.pyplot as plt
from ibllib.ephys import neuropixel
from one.api import ONE
from pathlib import Path
import brainbox.plot as bbplot
# from brainbox.io.one import load_spike_sorting_with_channel
from brainbox.io.one import load_spike_sorting_fast
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
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

fig_path = Path.home().joinpath('Desktop/Overview_plot_BWM/Repeatedsite/Raster_Fig1_V3')
fig_path.mkdir(parents=True, exist_ok=True)

# Select pids to plot
pids = ['4b93a168-0f3b-4124-88fa-a57046ca70e1',  # BAD - Low yield
        '57656bee-e32e-4848-b924-0f6f18cfdfb1',
        'c4f6665f-8be5-476b-a6e8-d81eeae9279d',
        '523f8301-4f56-4faf-ab33-a9ff11331118',  # GOOD
        '63517fd4-ece1-49eb-9259-371dc30b1dd6',
        'a12c8ae8-d5ad-4d15-b805-436ad23e5ad1',
        'c07d13ed-e387-4457-8e33-1d16aed3fd92',
        'f86e9571-63ff-4116-9c40-aa44d57d2da9',
        'f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c',
        '7cbecb3f-6a8a-48e5-a3be-8f7a762b5a04',  # BAD - Miss target
        '8ca1a850-26ef-42be-8b28-c2e2d12f06d6',
        '63a32e5c-f63a-450d-85cb-140947b67eaf']

len_pids = len(pids)

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
    axs[2].set_xlim([0, 60*60])  # display 1 hour

    # Save plot
    fig.tight_layout()
    fname = fig_path.joinpath(f'{pid}_RasterHist.pdf')
    plt.savefig(fname=fname)
    plt.close(fig)
