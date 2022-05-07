## Installation
To generate figures that have 3D visualisations of the brain the following extra packages must be installed 
```
conda activate ibl_repro_ephys
pip install mayavi
pip install pyqt5
```

## Figure 1

Figure 1 can be reproduced by executing figure1_run.py file 
```
python figure1_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from figure1.figure1_run import run_figure1
one = ONE()
run_figure1(one)

```
## Info
Currently the code only creates the individual panels for the figures and does not construct the whole figures shown in
the paper
