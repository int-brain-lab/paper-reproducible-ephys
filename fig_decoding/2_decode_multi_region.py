"""Example script for running multi-region reduced-rank model w/o hyperparameter sweep.
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from utils.data_loader_utils import MultiRegionDataModule
from models.decoders import MultiRegionReducedRankDecoder
from utils.eval_utils import eval_multi_region_model
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

"""
-----------
USER INPUTS
-----------
"""
ap = argparse.ArgumentParser()
ap.add_argument(
    "--target", type=str, default="choice", 
    choices=["choice", "stimside", "reward", "wheel-speed", "whisker-motion-energy"]
)
ap.add_argument("--query_region", nargs="+", default=["PO", "LP", "DG", "CA1", "VISa"])
ap.add_argument("--method", type=str, default="reduced_rank", choices=["reduced_rank"])
ap.add_argument("--fold_idx", type=int, default=1)
ap.add_argument("--num_epochs", type=int, default=2000)
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
args = ap.parse_args()


"""
-------
CONFIGS
-------
"""
kwargs = {"model": "include:src/configs/decoder.yaml"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/decoder.yaml", config)

if args.target in ["wheel-speed", "whisker-motion-energy", "pupil-diameter"]:
    config = update_config("src/configs/reg_trainer.yaml", config)
    weight_decay = 0.01
    temporal_rank = 4
    global_basis_rank = 12
elif args.target in ["choice", "stimside", "reward"]:
    config = update_config("src/configs/clf_trainer.yaml", config)
    weight_decay = 1
    temporal_rank = 2
    global_basis_rank = 5
else:
    raise NotImplementedError

set_seed(config.seed)

config["dirs"]["data_dir"] = Path(args.base_path)/config.dirs.data_dir/f"fold_{args.fold_idx}"
save_path = Path(args.base_path)/config.dirs.output_dir/args.target/f'multi-region-{args.method}'/f"fold_{args.fold_idx}"
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.target/f'multi-region-{args.method}'/f"fold_{args.fold_idx}"
os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)

model_class = args.method
query_region = args.query_region


"""
---------
LOAD DATA
---------
"""
eids = [fname for fname in os.listdir(config.dirs.data_dir) if fname != "downloads"]
print('---------------------------------------------')
print(f'Decode {args.target} from {len(eids)} sessions:')
print(eids)


"""
--------
DECODING
--------
"""
print('---------------------------------------------')
print(f'Launch multi-region {model_class} decoder:')

# set up model configs
base_config = config.copy()
base_config['target'] = args.target
base_config['region'] = 'all' 
base_config['query_region'] = query_region
base_config['training']['device'] = torch.device(
    'cuda' if np.logical_and(torch.cuda.is_available(), config.training.device=='gpu') else 'cpu'
)

base_config['optimizer']['lr'] = 5e-4
base_config['optimizer']['weight_decay'] = weight_decay

if model_class == "reduced_rank":
    base_config['temporal_rank'] = temporal_rank
    base_config['global_basis_rank'] = global_basis_rank
    base_config['tuner']['num_epochs'] = 500 
    base_config['training']['num_epochs'] = args.num_epochs 
else:
    raise NotImplementedError

print("Model config:")
print(base_config)

# set up trainer
checkpoint_callback = ModelCheckpoint(
    monitor="val_metric", mode="max", save_top_k=1, filename="best_model",
)
trainer = Trainer(
    max_epochs=config.training.num_epochs,
    check_val_every_n_epoch=1,
    callbacks=[checkpoint_callback], 
    enable_progress_bar=config.training.enable_progress_bar,
    fast_dev_run=False
)

# set up data loader
configs = []
for eid in eids:
    config = base_config.copy()
    config['eid'] = eid
    configs.append(config)
dm = MultiRegionDataModule(eids, configs)
dm.list_regions()  # check all available regions
regions_dict = dm.regions_dict

configs = []
for eid in eids:
    for region in query_region:
        # only load data from sessions containing this region
        if region in regions_dict[eid]:
            config = base_config.copy()
            config['eid'] = eid
            config['region'] = region
            configs.append(config)
dm = MultiRegionDataModule(eids, configs)
dm.setup()

np.save(save_path/'configs.npy', dm.configs)

# init and train model
base_config = dm.configs[0].copy()
base_config['n_units'], base_config['eid_region_to_indx'] = [], {}
for eid in eids:
    base_config['eid_region_to_indx'][eid] = {}
# build a dict indexing each session-region combination
for idx, _config in enumerate(dm.configs):
    base_config['n_units'].append(_config['n_units'])
    base_config['eid_region_to_indx'][_config['eid']][_config['region']] = idx
# build a dict indexing each brain region
base_config['region_to_indx'] = {r: i for i,r in enumerate(query_region)}
base_config['n_regions'] = len(query_region)

print("Index for region and session:")
print(base_config['eid_region_to_indx'])
 
if model_class == "reduced_rank":
    model = MultiRegionReducedRankDecoder(base_config)
else:
    raise NotImplementedError
trainer.fit(model, datamodule=dm)

best_model_path = checkpoint_callback.best_model_path
print("Best model path: ", best_model_path)
trainer.test(datamodule=dm, ckpt_path=best_model_path)

model = MultiRegionReducedRankDecoder.load_from_checkpoint(best_model_path, config=base_config)

"""
----------
EVALUATION
----------
"""
metric_dict, chance_metric_dict, test_pred_dict, test_y_dict = eval_multi_region_model(
    dm.train, dm.test, model, 
    target=base_config['model']['target'], 
    beh_name=args.target,
    save_path=save_path,
    data_dir=config["dirs"]["data_dir"],
    load_local=True,
    huggingface_org="neurofm123",
    all_regions=query_region, 
    configs=dm.configs,
)
print("Decoding results for each session and region:")
print(metric_dict)
print("Chance-level decoding results:")
print(chance_metric_dict)
    
for region in metric_dict.keys():
    for eid in metric_dict[region].keys():
        res_dict = {
            'test_metric': metric_dict[region][eid], 
            'test_chance_metric': chance_metric_dict[region][eid], 
            'test_pred': test_pred_dict[region][eid], 
            'test_y': test_y_dict[region][eid]
        }
        os.makedirs(save_path/region, exist_ok=True)
        np.save(save_path/region/f'{eid}.npy', res_dict)
        
