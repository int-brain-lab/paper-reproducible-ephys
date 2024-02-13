import colorsys
from iblatlas.regions import BrainRegions
import numpy as np

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


br = BrainRegions()

def get_colors_region(region):
    """
    :param region: Cosmos acronym
    :returns: colors, colors_translucent
    """
    region_idx = br.acronym2index([region])[1][0]
    region_rgb = br.rgb[region_idx][0]
    colors = [region_rgb / 255., scale_lightness(region_rgb/255., 1.2)]
    colors_translucent = [np.array(list(colors[0]) + [0.75]), np.array(list(colors[1]) + [0.75])]
    return colors, colors_translucent






