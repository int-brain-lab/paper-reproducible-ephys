## Figure 7

The data for Figure 7 can be reproduced by executing figure7_run.py file 
```
python figure6_run.py
```

or by using the following code snippet in an ipython terminal

```python

from one.api import ONE
from figure6.figure6_run import run_figure6
one = ONE()
run_figure6(one)

# To also generate the supplementary figures
run_figure6(one, supplementary=True)

```

## Info
Code to reproduce the plots is not yet available