## Installation
In order to run code for this figure pytorch must be installed. This figure can only be generated on a computer with
an nvidia driver installed.
To install pytorch for your OS please follow this link https://pytorch.org/ and refer to the `Install Pytorch` section

# Figures mtnn
The figures for the mtnn analysis can be reproduced by executing fig_mtnn_run.py file 
```
python fig_mtnn_run.py
```

or by using the following code snippet in an ipython terminal

```python

from one.api import ONE
from fig_mtnn.fig_mtnn_run import run_fig_mtnn
one = ONE()
run_fig_mtnn(one)
```

By default the data and trained models are downloaded and the figures reproduced. If you want to rerun the full analysis
the following can be run N.B Training takes a couple of days 
```python

from one.api import ONE
from fig_mtnn.fig_mtnn_run import run_fig_mtnn
one = ONE()
run_fig_mtnn(one, do_training=True)
```

