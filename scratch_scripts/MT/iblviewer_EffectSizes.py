#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marsa Taheri, edited from the ibl_points_neuron.py example
"""

import os
import numpy as np

from iblviewer.mouse_brain import MouseBrainViewer


if __name__ == '__main__':
    viewer = MouseBrainViewer()
    # viewer.initialize(resolution=50, mapping='Allen', add_atlas=True, add_dwi=True, 
    #                 dwi_color_map='Greys_r', embed_ui=True)
    viewer.initialize(resolution=25, embed_ui=True)

    # Load effect sizes (current file from Hyun)
    effect_size_dict = np.load('SavedFiles/effect_sizes_dict.npy', allow_pickle=True).tolist()

    # Extract xyz coordinates (convert from meters to microns) and the value 
    XYZs_all, ValuesPerKey = [], []
    for key in effect_size_dict.keys():
        XYZs_all.append(key[3])
        DictPerKey = effect_size_dict[key]
        ValuesPerKey.append(np.array(DictPerKey['wheel'])) #currently only one key (for one effect size category) is used
    ValuesPerKey = np.array(ValuesPerKey)
    XYZs = np.array(XYZs_all)*1e6
        
    #point_actors = []
    #viewer.plot.remove(point_actors, render=False)

    point_actors = viewer.add_points(XYZs, radius=16, values=ValuesPerKey, screen_space=False, 
                               noise_amount=0, min_v=ValuesPerKey.min(), max_v=ValuesPerKey.max(), color_map='viridis')
     #viridis, vlag, Accent, cool
    viewer.plot.add(point_actors)

    viewer.show().close()
  