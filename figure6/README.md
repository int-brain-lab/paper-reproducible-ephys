## Figure 6

Figure 6 can be reproduced by executing figure6_run.py file 
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