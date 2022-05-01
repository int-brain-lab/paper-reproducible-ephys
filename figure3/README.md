## Figure 3

Figure 3 can be reproduced by executing figure3_run.py file 
```
python figure3_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from figure3.figure3_run import run_figure3
one = ONE()
run_figure3(one)

# To also generate the supplementary figures
run_figure3(one, supplementary=True)

```
