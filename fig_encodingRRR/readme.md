## Figure linear encoding model
This code can be used to reproduce Figure 7.

In order to run code for this figure pytorch must be installed. This figure can only be generated on a computer with an nvidia driver installed. To install pytorch for your OS please follow this link https://pytorch.org/ and refer to the Install Pytorch section

#### Quick Start: Shortcut pipeline 
To quickly reproduce Figure 7 using preprocessed data:
1. run `python decoding_analysis.py`, the resulting figure will be saved in `./result/` directory. 

#### Full pipeline: Step-by-step workflow
Follow these steps to execute the full data preparation, model training, and decoding analysis pipeline:
1. Use `prepare_data.py` to download and preprocess the data from IBL API.
    - data will be saved in "./data/rep_ephys_22032024/downloaded/"
2. Train RRR encoding model using `train.py`.
    - trained model will be saved in f"./model/rep_ephys_22032024/{preprocess_kwargs}/"
3. Use `ensemble_data_for_decoding.py` to ensemble X (the 32-dimensional single-neuron response profile) and y (lab or area) for the decoding analysis
    - results saved in f"./result/data.npz"
4. Perform the decoding analysis using `decoding_analysis.py`.
    - figure saved in f"./result/"
