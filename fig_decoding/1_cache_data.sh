#!/bin/bash

BASE_PATH="XXX"   # set the save path (adjust as needed)
WORKING_DIR="XXX" # set the working directory (adjust as needed)

conda activate ibl_repro_ephys

cd $WORKING_DIR

for fold_idx in {1..5}
do
    python src/1_data_caching.py --base_path $BASE_PATH --fold_idx $fold_idx
done

conda deactivate


