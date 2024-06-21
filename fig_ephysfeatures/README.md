## Figure Ephys Features

This code can be used to reproduce Figure 3 in the paper.

The figures can be reproduced by executing the ephysfeatures_run.py file 
```
python ephysfeatures_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from fig_ephysfeatures.ephysfeatures_run import run_fig_ephysfeatures
one = ONE()
run_fig_ephysfeatures(one)

```

To generate supplementary material you can run
```python
run_fig_ephysfeatures(one, supplementary=True)
```


To find out where the data and figures have been stored locally you can do the following
```python
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path('fig_ephysfeatures'))
print(save_figure_path('fig_ephysfeatures'))
```

## Info
We have not released data to replicate the results for the bilateral recordings fig 3 supp 3