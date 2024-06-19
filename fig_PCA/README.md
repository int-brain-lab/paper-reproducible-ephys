## Figure PCA

This code can be used to reproduce Figure 5 in the paper

The figures can be reproduced by executing the fig_PCA_run.py file 
```
python fig_PCA_run.py
```

or by using the following code snippet in an ipython terminal

```python

from one.api import ONE
from fig_PCA.fig_PCA_run import run_fig_PCA
one = ONE()
run_fig_PCA(one)

```

To find out where the data and figures have been stored locally you can do the following
```python
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path('fig_PCA'))
print(save_figure_path('fig_PCA'))
```