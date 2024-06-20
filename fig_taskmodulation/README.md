## Figure Task Modulation

This code can be used to reproduce Figure 4 in the paper

The figures can be reproduced by executing the fig_taskmodulation_run.py file 

```
python fig_taskmodulation_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from fig_taskmodulation.fig_taskmodulation_run import run_fig_taskmodulation
one = ONE()
run_fig_taskmodulation(one)

```

To find out where the data and figures have been stored locally you can do the following
```python
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path('fig_taskmodulation'))
print(save_figure_path('fig_taskmodulation'))
```

## Info
The power analysis computation can take a while to run
