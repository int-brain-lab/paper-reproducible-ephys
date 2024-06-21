## Installation
To generate figures that have 3D visualisations of the brain the following extra packages must be installed 
```
conda activate ibl_repro_ephys
pip install mayavi
pip install pyqt5
```

## Figure Intro

This code can be used to reproduce Figure 1, Figure 1 Supp2 and Figure 1 Supp3  in the paper

The figures can be reproduced by executing the fig_intro_run.py file 
```
python fig_intro_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from fig_intro.fig_intro_run import run_fig_intro
one = ONE()
run_fig_intro(one)

```

To find out where the data and figures have been stored locally you can do the following
```python
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path('fig_intro'))
print(save_figure_path('fig_intro'))
```

## Info
Currently the code only creates the individual panels for the figures and does not construct the whole figures shown in
the paper. We have not released data for the poor quality examples in figure 1 supp 3 so this figure cannot be reproduced
