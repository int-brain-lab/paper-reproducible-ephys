
import sys

import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\noamroth\int-brain-lab\paper-reproducible-ephys')
from one.api import ONE
one = ONE()

from reproducible_ephys_functions import get_insertions, compute_metrics
ins = get_insertions(level=0, one=one, freeze='release_2022_11')
_ = compute_metrics(ins, one=one)

from reproducible_ephys_functions import filter_recordings
df = filter_recordings()
df = df[df['lab'] == 'steinmetzlab']


from one.api import ONE
from figure3.figure3_run import run_figure3
one = ONE()
run_figure3(one)
#%%
sys.path.append(r'C:\Users\noamroth\int-brain-lab\paper-reproducible-ephys')
from one.api import ONE
one = ONE()
from figure3.figure3_plot_functions import (panel_probe_lfp, panel_probe_neurons, panel_example,
                                            panel_permutation, panel_sankey, panel_decoding)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,1)

panel_sankey(fig, ax, one, freeze='release_2022_11')
plt.savefig(r'C:\Users\noamroth\int-brain-lab\paper-reproducible-ephys\scratch_scripts\NR\sankey.svg',dpi=500)
plt.show()

#%%
from reproducible_ephys_functions import query
allIns = query(behavior=False, n_trials=0, resolved=False, min_regions=0, exclude_critical=False, one=one)
nonCrit = query(behavior=False, n_trials=0, resolved=False, min_regions=0, exclude_critical=True, one=one)

crit = [x for x in allIns if x not in nonCrit]
crit_pids = [crit[i]['probe_insertion'] for i in range(len(crit))]