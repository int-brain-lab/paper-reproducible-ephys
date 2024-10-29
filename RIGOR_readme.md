In this repo we have provided code that allows you to run the RIGOR metrics on your own data. 

At the insertion level these include criteria on the rms of the AP band, the derivate of the LFP band and 
the neuron yield on the probe. Please refer to our paper for the code for the exact threshold values applied.

At the single neuron level we apply quality control metrics that include an amplitude criteria, a noise cutoff criteria
and sliding re criteria. The IBL spikesorting white paper contains more details about these single unit quality control
metrics.

To run these RIGOR metrics on your data you will need to provide,
1. The folder path containing your spikesorted data
2. The folder path containing your raw electrophysiology data (currently only supports spikeglx)
3. A folder to save the results to (should be different from the folders provided in 1 and 2)

If the lf.bin data has not been recorded (in the case of Neuropixel 2.0 probes), then short snippets of lfp
data will be extracted from the ap.bin file by applying a low pass filter.

To compute metrics you can use the following
```python
python RIGOR_script.py -r raw_data_path -s spikesorting_path -o output_path
```

In the terminal the RIGOR metrics at the insertion level will be displayed. In the output path you will
also find a plot called RIGOR_plot.png that will create an overview plot displaying some of the features on your 
recording.

If you pass in data that is 4 shank data, it will automatically detect the shanks and compute the RIGOR metrics
on each shank individually.
