import numpy as np
from ibllib.atlas.regions import BrainRegions
from neuropixel import SITES_COORDINATES
from matplotlib.image import NonUniformImage


def get_brain_boundaries(brain_regions, z, r=None):

    r = r or BrainRegions()
    all_levels = {}
    _brain_id = r.get(ids=brain_regions['id'])
    level = 10
    while level > 0:
        brain_regions = r.get(ids=_brain_id['id'])
        level = np.nanmax(brain_regions.level).astype(int)
        all_levels[f'level_{level}'] = brain_regions['acronym']
        idx = np.where(brain_regions['level'] == level)[0]
        _brain_id = brain_regions
        _brain_id['id'][idx] = _brain_id['parent'][idx]

    boundaries = []
    colours = []
    regions = []
    void = np.where(all_levels['level_3'] == 'void')[0]
    if len(void) > 2:
        boundaries.append(z[void[0]])
        idx = np.where(r.acronym == 'VIS')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        regions.append('void-VIS')
    ctx = np.where(all_levels['level_5'] == 'Isocortex')[0]
    if len(ctx) > 2:
        boundaries.append(z[ctx[0]])
        idx = np.where(r.acronym == 'VIS')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        regions.append('VIS-HPF')
    hip = np.where(all_levels['level_5'] == 'HPF')[0]
    if len(hip) > 2:
        boundaries.append(z[hip[-1]])
        boundaries.append(z[hip[0]])
        idx = np.where(r.acronym == 'HPF')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        colours.append(rgb)
        regions.append('HPF-DG')
    thal = np.where(all_levels['level_2'] == 'BS')[0]
    if len(thal) > 2:
        boundaries.append(z[thal[-1]])
        idx = np.where(r.acronym == 'TH')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        regions.append('DG-TH')

    return boundaries, colours, regions


def plot_probe(data, z, ax, clim=[0.1, 0.9], normalize=True, cmap=None):
    bnk_x, bnk_y, bnk_data = arrange_channels2banks(data, z)
    for x, y, dat in zip(bnk_x, bnk_y, bnk_data):
        im = NonUniformImage(ax, interpolation='nearest', cmap=cmap)
        if normalize:
            levels = np.nanquantile(bnk_data[0], clim)
            im.set_clim(levels[0], levels[1])
        else:
            im.set_clim(clim)
        im.set_data(x, y, dat.T)
        ax.images.append(im)
    ax.set_xlim(0, 4.5)
    return im


def arrange_channels2banks(data, y):
    bnk_data = []
    bnk_y = []
    bnk_x = []
    for iX, x in enumerate(np.unique(SITES_COORDINATES[:, 0])):
        bnk_idx = np.where(SITES_COORDINATES[:, 0] == x)[0]
        bnk_vals = data[bnk_idx]
        bnk_vals = np.insert(bnk_vals, 0, np.nan)
        bnk_vals = np.append(bnk_vals, np.nan)
        bnk_vals = bnk_vals[:, np.newaxis].T
        bnk_vals = np.insert(bnk_vals, 0, np.full(bnk_vals.shape[1], np.nan), axis=0)
        bnk_vals = np.append(bnk_vals, np.full((1, bnk_vals.shape[1]), np.nan), axis=0)
        bnk_data.append(bnk_vals)

        y_pos = y[bnk_idx]
        y_pos = np.insert(y_pos, 0, y_pos[0] - np.abs(y_pos[2] - y_pos[0]))
        y_pos = np.append(y_pos, y_pos[-1] + np.abs(y_pos[-3] - y_pos[-1]))
        bnk_y.append(y_pos)

        x = np.arange(iX, iX + 3)
        bnk_x.append(x)

    return bnk_x, bnk_y, bnk_data
