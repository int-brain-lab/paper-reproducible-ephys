## Installation
In order to run code for this figure pytorch must be installed. This figure can only be generated on a computer with
an nvidia driver installed.
To install pytorch for your OS please follow this link https://pytorch.org/ and refer to the `Install Pytorch` section

# Figure 9 and 10
Figures 9 and 10 can be reproduced by executing figure9_10_run.py file 
```
python figure9_10_run.py
```

or by using the following code snippet in an ipython terminal

```python

from one.api import ONE
from figure9_10.figure9_10_run import run_figure9_10
one = ONE()
run_figure9_10(one)
```

By default the data and trained models are downloaded and the figures reproduced. If you want to rerun the full analysis
the following can be run N.B Training takes a couple of days 
```python

from one.api import ONE
from figure9_10.figure9_10_run import run_figure9_10
one = ONE()
run_figure9_10(one, do_training=True)
```

