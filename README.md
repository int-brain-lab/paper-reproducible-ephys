This repository contains code to reproduce figures from the IBL reproducible ephys paper
 
# Installation
## Making a new python environment (optional)

Install [Anaconda](https://www.anaconda.com/distribution/#download-section) and [git](https://git-scm.com/downloads), 
and follow their installer instructions to add each to the system path

Create new python environment
```
conda create --name ibl_repro_ephys python=3.9
```
Activate environment
```
conda activate ibl_repro_ephys
```

## Downloading and Installing repo

Clone the repo 
```
git clone https://github.com/int-brain-lab/paper-reproducible-ephys.git
```

Navigate to repo
```
cd paper-reproducible-ephys
```

Install requirements and repo
```
pip install -e .
```

# Setting up ONE credentials
Open an ipython terminal
```
from one.api import ONE
pw = 'international'
one = ONE(silent=True, password=pw)
```

# Setting up saving scripts
By default data and figures will be saved into a folder with the figure name e.g figure1. 
To find this location on you computer (for example for figure 1) you can type the 
```
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path(figure='figure1'))
print(save_figure_path(figure='figure1'))
```

If you want to override the location where the data and figures are saved you can create a script in the repo directory,
that is called reproducible_ephys_paths.py and add the following following: 

FIG_PATH = '/path/where/to/save/your/figures/'
DATA_PATH = '/path/where/to/save/your/data/

# Getting Started
To get started you will need to execute the code reproducible_ephys_run.py which will download the data and compute
exclusion criteria for the insertions 
```
python reproducible_ephys_run.py
```
This can also be run from a python terminal
or by using the following code snippet in an ipython terminal
```python

from one.api import ONE
from reproducible_ephys_run import run_repro_ephys_metrics
one = ONE()
run_repro_ephys_metrics(one)

```

# Running figures
For instructions on how to run code for each figure please refer to README file in each figure sub-folder
