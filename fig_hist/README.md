## Figure Histology

This code can be used to reproduce Figure 2 in the paper

The figures can be reproduced by executing the fig_hist_run.py file 
```
python fig_hist_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from fig_hist.fig_hist_run import run_fig_hist
one = ONE()
run_fig_hist(one)

```

To find out where the data and figures have been stored locally you can do the following
```python
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path('fig_hist'))
print(save_figure_path('fig_hist'))
```

## Info
Currently raw histology data for panel B in figure 2 and figure 2 supplementary 1 has not been released and so these
figures can not be reproduced programatically.
