
### Setup environment

```
git clone https://github.com/yzhang511/neural_decoding.git
cd neural_decoding
git checkout repro-ephys
conda env create -f env.yaml
conda activate ibl_repro_ephys
```

### Create datasets

Adjust the necessary path: 
- `BASE_PATH`: Where to save the outputs
- `WORKING_DIR`: The path to the neural_decoding code

```
source 0_filter_data.sh
source 1_cache_data.sh
```

### Fit decoding models

- Adjust the necessary path.
- Set the target variable to decode (choice, stimside, reward, wheel-speed).

```
source 2_decode_multi_region.sh choice
source 2_decode_multi_region.sh stimside
source 2_decode_multi_region.sh reward
source 2_decode_multi_region.sh wheel-speed
```

NOTE: The above code takes a long time so that you may want to change the `.sh` script and submit to slurm.

### Run decoding analyses

Analyze decoding results and save intermediate outputs for plotting:
- `--base_path`: Where to load decoding results from the last step

```
python 3_analyze_results.py --base_path XXX
```

### Plot

```
python 4_generate_plot.py
```



