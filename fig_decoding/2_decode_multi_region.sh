#!/bin/bash

BASE_PATH="XXX"   # set the save path (adjust as needed)
WORKING_DIR="XXX" # set the working directory (adjust as needed)

target=${1}

conda activate ibl_repro_ephys
cd $WORKING_DIR

for fold_idx in {1..5}
do
python src/2_decode_multi_region.py --target $target --query_region PO LP DG CA1 VISa --fold_idx $fold_idx --num_epochs 2000 --base_path $BASE_PATH
done

conda deactivate
