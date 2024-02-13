## Figure 3

Figure 3 can be reproduced by executing ephysfeatures_run.py file 
```
python ephysfeatures_run.py
```

or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from fig_ephysfeatures.ephysfeatures_run import run_fig_ephysfeatures
one = ONE(base_url='https://openalyx.internationalbrainlab.org')
run_fig_ephysfeatures(one)

```

To generate supplementary material you can run
```python
run_fig_ephysfeatures(one, supplementary=True)
```
