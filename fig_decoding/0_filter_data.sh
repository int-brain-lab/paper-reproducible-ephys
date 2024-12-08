#!/bin/bash

BASE_PATH="XXX"   # set the save path (adjust as needed)
WORKING_DIR="XXX" # set the working directory (adjust as needed)

conda activate decoding

cd $WORKING_DIR

python src/0_data_filtering.py --base_path $BASE_PATH

conda deactivate


