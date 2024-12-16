## Figure Data Quality

This code can be used to reproduce Figure 1f, and Figure 1 Supplements 3 and 4.

The figures can be reproduced by executing the fig_data_quality_run.py file 
```
python fig_data_quality_run.py
```

or by using the following code snippet in an ipython terminal

```python

from fig_data_quality.fig_data_quality_run import run_fig_data_quality
run_fig_data_quality()

```

To find out where the data and figures have been stored locally you can do the following
```python
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path('fig_data_quality'))
print(save_figure_path('data_quality'))
```