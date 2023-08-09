## Figure 4

Figure 4 can be reproduced by executing fig_taskmodulation_run.py file 
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