This repository contains code to reproduce figures from the 
[IBL reproducible ephys paper] TODO NEW LINK TO PAPER
 
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

# Configuration
## Setting up ONE credentials
Open an ipython terminal
```
from one.api import ONE
pw = 'international'
one = ONE(silent=True, password=pw)
```

## Setting up saving scripts
By default data and figures will be saved into a folder with the figure name e.g fig_hist. 
To find this location on you computer (for example for figure 1) you can type the 
```
from reproducible_ephys_functions import save_data_path, save_figure_path
print(save_data_path(figure='fig_hist'))
print(save_figure_path(figure='fig_hist'))
```

If you want to override the location where the data and figures are saved you can create a script in the repo directory,
that is called reproducible_ephys_paths.py and add the following: 

FIG_PATH = '/path/where/to/save/your/figures/'

DATA_PATH = '/path/where/to/save/your/data/

# Getting Started

## Reproducing the figures
In each figure subfolder there is a read me that contains instructions for how to replicate the analysis and 
generate the figures in the paper.

The subfolders correspond to the following figures
* Figure 1 - fig_intro, fig_data_quality
* Figure 2 - fig_hist
* Figure 3 - fig_ephysfeatures
* Figure 4 - fig_taskmodulation
* Figure 5 - fig_PCA
* Figure 6 - fig_spatial
* Figure 7 - fig_mtnn

## Finding the insertions used for analysis
The list of insertions probe insertions considered for analysis in this version of the paper 
can be found in the following way
```python
from one.api import ONE
from reproducible_ephys_functions import get_insertions

one = ONE()
insertions = get_insertions(level=0, one=one, freeze='freeze_2024_03')
```


