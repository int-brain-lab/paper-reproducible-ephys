## Figure Spatial

This code can be used to reproduce Figure 6 in the paper.

The data for the figures can be reproduced by executing the fig_spatial_run.py file
```
python fig_spatial_run.py
```

or by using the following code snippet in an ipython terminal

```python

from one.api import ONE
from fig_spatial.fig_spatial_run import run_fig_spatial
one = ONE()
run_fig_spatial(one)
```

The code for generating the figures is currently in Matlab (except for the time course plots of the Fano Factor supplementary figure). 
To generate the figures in Matlab you will need to manually change the `data_path` and `save_path` in the figure `fig_spatial_plot_data_matlab.m` 
code to the paths that you get out from the following code

```python

from reproducible_ephys_functions import save_data_path, save_figure_path
print(f'data_path={str(save_data_path(figure="fig_spatial"))}')
print(f'save_path={str(save_figure_path(figure="fig_spatial"))}')
```

To generate the figures in matlab you can then run
```commandline
fig_spatial_plot_data_matlab.m
```

To generate the time course plots for the supplementary figure on the Fano Factor you can run
```python
python fig_spatial_plot_data.py
```

## Info
Only individual panels are made for these figures, the full figures have been created manually post-hoc

