## Figure 7

The data for Figure 7 can be reproduced by executing figure7_run.py file 
```
python figure7_run.py
```

or by using the following code snippet in an ipython terminal

```python

from one.api import ONE
from figure7.figure7_run import run_figure7
one = ONE()
run_figure7(one)
```

The code for generating the figures is currently in matlab. To generate the figures in matlab you will need to 
manually change the `data_path` and `save_path` in the figure `figure7_plot_data_matlab.m` code 
to the paths that you get out from the following code
```python

from reproducible_ephys_functions import save_data_path, save_figure_path
print(f'data_path={str(save_data_path(figure="figure7"))}')
print(f'save_path={str(save_figure_path(figure="figure7"))}')
```

# Info 
Currently the plots for supplementary figure are not arranged into a single figure

