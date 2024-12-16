## Installation
To generate figures that have 3D visualisations of the brain the following extra packages must be installed 
```
conda activate ibl_repro_ephys
pip install mayavi
pip install pyqt5
```

## Figure Intro

This code can be used to reproduce some panels of Figure 1

The figures can be reproduced by executing the fig_intro_run.py file 
```
python fig_intro_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from fig_intro.fig_intro_run import run_fig_intro
one = ONE()
run_fi****g_intro(one)

```

To find out where the data and figures have been stored locally you can do the following
```python
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path('fig_intro'))
print(save_figure_path('fig_intro'))
```

## Info
We have not released data for the poor quality examples in figure 1 supp 2 so this figure cannot be reproduced.
