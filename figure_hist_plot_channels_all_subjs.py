#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This Python Script generates a figure containing all subject histology plots
coronal/sagittal through the repeated site histology data, and plots the
probe channels onto this histology data.

@author: sjwest
"""

def print_path():
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    print(path)


def plot_all_channels(output='figure_histology_supp', colour='w'):
    '''Plot all subjects CORONAL & SAGITTAL histology and channels for repeated site
    
    Gets the subjects to plot from the load_channels_data() function in 
    probe_geometry_data module.
    
    Plots all coronal and all sagittal data in one large figure.
    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    from one.api import ONE
    from ibllib.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import ibllib.atlas as atlas
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # connect to ibl server
    one = ONE()
    
    # load channels data for repeated site from local cache
    ch_data = fhd.get_channels_data('-2243_-2000')
    
    # get subject IDs as list
    ids = list(dict.fromkeys(ch_data['subject']))
    
    # len(ids) is 69
     # generate a figure with axes - 4 rows, 17 cols (max 68)
      # coronal and sagittal
    nRows = 5
    nCols = 14
    
    # some parameters for fig gen:
    gr_percentile_min=0.5
    gr_percentile_max=99.9
    rd_percentile_min=1
    rd_percentile_max=99.9
    font_size = 6
    label_size = 7
    
    # create figures
    figcor = plt.figure(figsize=(60, 20), dpi=72)
    figsag = plt.figure(figsize=(60, 20), dpi=72)
    
    # loop through all ids and generate tilted slice plots with recording sites plotted
    row_index = -1
    col_index = 0
    for i in range(0, len(ids)):
        
        # keep track of row/col
        if(i%nCols == 0):
            row_index=row_index+1
        col_index = i%nCols
        
        print('row ', row_index)
        print('col', col_index)
        
        subject_ID = ids[i]
        sub_index = ch_data['subject']==subject_ID
        lab = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'lab']
        eid = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'eid']
        probe = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'probe']
    
        # Fetch trajectory DICT - 'Histology track'
        traj = one.alyx.rest('trajectories', 'list', session=eid, probe=probe, provenance='Histology track')
    
        # Create insertion OBJECT from trajectory DICT
        ins = Insertion.from_dict(traj[0])
        
        axc = figcor.add_subplot(nRows, nCols, (i+1))
        axs = figsag.add_subplot(nRows, nCols, (i+1))
        
        # get the insertion for id 0 - NYU-12
        print( str(i) + " : " + subject_ID)
        
        # labels for axes
        axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
        
        hist_paths = hist.download_histology_data(subject_ID, lab)
        
        # create the brain atlases from the data
        ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
        ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
        
        #gr_thal_roi = ba_gr.image[150:200, 178:278, 100:200]
        # in Z slices run from ANTERIOR to POSTERIOR (528-150,200)
        gr_thal_roi = ba_gr.image[328:378, 178:278, 100:200] # isolate large slice over thalamus for max pixel value
        
        # CORONAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 1, volume = ba_rd.image)
        
        #gr_tslice_roi = gr_tslice[120:240, 150:300] # isolate large slice over thalamus for max pixel value
        #rd_tslice_roi = rd_tslice[120:240, 150:300]
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and gr_thal_roi max to scale autofl.
          # using rd_tslice min and percentile (99.99 default) to scale CM-DiI
        gr_in = np.interp(gr_tslice, (np.percentile(gr_tslice, gr_percentile_min), 
                                      np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, (np.percentile(rd_tslice, rd_percentile_min), 
                                      np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to imshow()
        axc.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        sec_ax = axc.secondary_yaxis('right', functions=(
                            lambda x: x * ab[0] + ab[1],
                            lambda y: (y - ab[1]) / ab[0]))
        
        axc.set_xlabel(axis_labels[0], fontsize=font_size)
        axc.set_ylabel(axis_labels[1], fontsize=font_size)
        sec_ax.set_ylabel(axis_labels[2], fontsize=font_size)
        
        axc.tick_params(axis='x', labelrotation = 90)
        
        axc.tick_params(axis='x', labelsize = label_size)
        axc.tick_params(axis='y', labelsize = label_size)
        sec_ax.tick_params(axis='y', labelsize = label_size)
        
        # SAGITTAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 0, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 0, volume = ba_rd.image)
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and max to scale the image
           # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in/1.5!
        #gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
        #rd_in = np.interp(rd_tslice, (gr_tslice.min(), gr_tslice.max()/1.5), (0, 255))
        gr_in = np.interp(gr_tslice, 
                          (np.percentile(gr_tslice, gr_percentile_min), 
                              np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, 
                          (np.percentile(rd_tslice, rd_percentile_min), 
                              np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to axs via imshow()
        axs.imshow(Zt, interpolation='none', aspect='auto', 
                    extent=np.r_[width, height], cmap=cmap, 
                    vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        #start = ins.xyz[:, 1] * 1e6
        #end = ins.xyz[:, 2] * 1e6
        #xCoords = np.array([start[0], end[0]])
        
        sec_ax = axs.secondary_yaxis('right', functions=(
                            lambda x: x * ab[0] + ab[1],
                            lambda y: (y - ab[1]) / ab[0]))
        
        axs.set_xlabel(axis_labels[2], fontsize=font_size)
        axs.set_ylabel(axis_labels[1], fontsize=font_size)
        sec_ax.set_ylabel(axis_labels[0], fontsize=font_size)
        
        axs.tick_params(axis='x', labelrotation = 90)
        
        axs.tick_params(axis='x', labelsize = label_size)
        axs.tick_params(axis='y', labelsize = label_size)
        sec_ax.tick_params(axis='y', labelsize = label_size)
        
        
        # add a line of the Insertion object onto axc (cax - coronal)
         # plotting PLANNED insertion 
        #axc.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        #axs.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        
        # crop coronal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 0]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 0]) *1e6 + 1000
        
        axc.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axc.axes.set_aspect('equal')
        
        # crop sagittal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 1]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 1]) *1e6 + 1000
        
        axs.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axs.axes.set_aspect('equal')
        
        
        # plot channels on each graph
        
        # subset the data_frame to subject
        subj_frame = ch_data[ch_data['subject'] == subject_ID]
        
        # retrieve the location in XYZ
        locX = subj_frame['chan_loc_x'].values
        locY = subj_frame['chan_loc_y'].values
        locZ = subj_frame['chan_loc_z'].values
        
        # plot channels as circles at half the dpi
         # this gives channel coords that are just about separate in the figure!
        axc.plot(locX * 1e6, locZ * 1e6,  marker='o',
                 ms=(72./figcor.dpi)/2, mew=0, 
            color=colour, linestyle="", lw=0)
        
        axs.plot(locY * 1e6, locZ * 1e6, marker='o',
                 ms=(72./figsag.dpi)/2, mew=0, 
            color=colour, linestyle="", lw=0)
        
        if col_index != 0:
            # remove the primary y axes
            print('  remove 1st y..')
            axc.get_yaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
        
    
    # adjust spacing
    wspace = 0.3   # the amount of width reserved for blank space between subplots
      # gives the tightest layout without overlap between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots
    
    figcor.subplots_adjust(wspace, hspace)
    figsag.subplots_adjust(wspace, hspace)
    
    # save to output
    figcor.savefig( str(Path(OUTPUT, 'all_channels_subj_hist_coronal.svg')), bbox_inches="tight" )
    figsag.savefig( str(Path(OUTPUT, 'all_channels_subj_hist_sagittal.svg')), bbox_inches="tight" )
    



def plot_channels_n3(output='figure_histology', remove_axes=True, colour='w'):
    '''Plot three subject, CORONAL & SAGITTAL histology and channels for repeated site
    
    Gets the subjects to plot from the load_channels_data() function in 
    probe_geometry_data module.
    
    Plots all coronal and all sagittal data in one large figure.
    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    from one.api import ONE
    from ibllib.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import ibllib.atlas as atlas
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # connect to ibl server
    one = ONE()
    
    # load channels data for repeated site from local cache
    ch_data = fhd.get_channels_data('-2243_-2000')
    
    # get subject IDs as list
    ids = list(dict.fromkeys(ch_data['subject']))
    
    # get set of ids - one per lab
    
    # get all subjects dict for BWM project
    subjs = one.alyx.rest('subjects', 'list', project='ibl_neuropixel_brainwide_01')
    
    # get all labs for ids
    labs = list()
    for s in ids:
        for m in subjs:
            if m['nickname'] == s:
                labs.append(m['lab'])
    
    # get set of labs - remove dups
    labs_set = set(labs)
    
    # get set of ids where LAST instance of lab from labs_set
    ids_sub = [None] * len(labs_set)
    for j, l in enumerate(labs_set):
        #print(j, l)
        for i, d in enumerate(ids):
            if labs[i] == l:
                ids_sub[j] = d
    
    # just set ids_sub to ids
    ids = ids_sub
    
    # set ids to subset of 3
    #ids = [ids[4], ids[5], ids[6]]
    
    # manually set the IDs
    #ids = ['CSH_ZAD_026', 'KS052', 'SWC_058']
    ids = ['CSH_ZAD_026', 'KS045', 'CSHL054']
    
    # ids is 3
     # generate a figure with axes - 2 rows, 5 cols (max 10)
      # coronal and sagittal
    nRows = 1
    nCols = 3
    
    # some parameters for fig gen:
    gr_percentile_min=0.5
    gr_percentile_max=99.9
    rd_percentile_min=1
    rd_percentile_max=99.9
    font_size = 6
    label_size = 7
    
    # create figures
    #figcor = plt.figure(figsize=(20, 10), dpi=72)
    #figsag = plt.figure(figsize=(20, 10), dpi=72)
    figcor = plt.figure()
    figsag = plt.figure()
    
    # reset the sizes
    figcor.set_size_inches(3, 2.15)
    figsag.set_size_inches(3, 2.15)
    
    # loop through all ids and generate tilted slice plots with recording sites plotted
    row_index = -1
    col_index = 0
    for i in range(0, len(ids)):
        
        # keep track of row/col
        if(i%nCols == 0):
            row_index=row_index+1
        col_index = i%nCols
        
        print('row ', row_index)
        print('col', col_index)
        
        subject_ID = ids[i]
        sub_index = ch_data['subject']==subject_ID
        lab = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'lab']
        eid = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'eid']
        probe = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'probe']
    
        # Fetch trajectory DICT - 'Histology track'
        traj = one.alyx.rest('trajectories', 'list', session=eid, probe=probe, provenance='Histology track')
    
        # Create insertion OBJECT from trajectory DICT
        ins = Insertion.from_dict(traj[0])
        
        axc = figcor.add_subplot(nRows, nCols, (i+1))
        axs = figsag.add_subplot(nRows, nCols, (i+1))
        
        # get the insertion for id 0 - NYU-12
        print( str(i) + " : " + subject_ID)
        
        # labels for axes
        axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
        
        hist_paths = hist.download_histology_data(subject_ID, lab)
        
        # create the brain atlases from the data
        ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
        ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
        
        #gr_thal_roi = ba_gr.image[150:200, 178:278, 100:200]
        # in Z slices run from ANTERIOR to POSTERIOR (528-150,200)
        gr_thal_roi = ba_gr.image[328:378, 178:278, 100:200] # isolate large slice over thalamus for max pixel value
        
        # CORONAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 1, volume = ba_rd.image)
        
        #gr_tslice_roi = gr_tslice[120:240, 150:300] # isolate large slice over thalamus for max pixel value
        #rd_tslice_roi = rd_tslice[120:240, 150:300]
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and gr_thal_roi max to scale autofl.
          # using rd_tslice min and percentile (99.99 default) to scale CM-DiI
        gr_in = np.interp(gr_tslice, (np.percentile(gr_tslice, gr_percentile_min), 
                                      np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, (np.percentile(rd_tslice, rd_percentile_min), 
                                      np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to imshow()
        axc.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        if remove_axes:
            axc.set_axis_off()
        else:
            sec_ax = axc.secondary_yaxis('right', functions=(
                                lambda x: x * ab[0] + ab[1],
                                lambda y: (y - ab[1]) / ab[0]))
            
            axc.set_xlabel(axis_labels[0], fontsize=font_size)
            axc.set_ylabel(axis_labels[1], fontsize=font_size)
            sec_ax.set_ylabel(axis_labels[2], fontsize=font_size)
            
            axc.tick_params(axis='x', labelrotation = 90)
            
            axc.tick_params(axis='x', labelsize = label_size)
            axc.tick_params(axis='y', labelsize = label_size)
            sec_ax.tick_params(axis='y', labelsize = label_size)
            
        
        # SAGITTAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 0, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 0, volume = ba_rd.image)
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and max to scale the image
           # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in/1.5!
        #gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
        #rd_in = np.interp(rd_tslice, (gr_tslice.min(), gr_tslice.max()/1.5), (0, 255))
        gr_in = np.interp(gr_tslice, 
                          (np.percentile(gr_tslice, gr_percentile_min), 
                              np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, 
                          (np.percentile(rd_tslice, rd_percentile_min), 
                              np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to axs via imshow()
        axs.imshow(Zt, interpolation='none', aspect='auto', 
                    extent=np.r_[width, height], cmap=cmap, 
                    vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        #start = ins.xyz[:, 1] * 1e6
        #end = ins.xyz[:, 2] * 1e6
        #xCoords = np.array([start[0], end[0]])
        if remove_axes:
            axs.set_axis_off()
            
        else:
            sec_ax = axs.secondary_yaxis('right', functions=(
                                lambda x: x * ab[0] + ab[1],
                                lambda y: (y - ab[1]) / ab[0]))
            
            axs.set_xlabel(axis_labels[2], fontsize=font_size)
            axs.set_ylabel(axis_labels[1], fontsize=font_size)
            sec_ax.set_ylabel(axis_labels[0], fontsize=font_size)
            
            axs.tick_params(axis='x', labelrotation = 90)
            
            axs.tick_params(axis='x', labelsize = label_size)
            axs.tick_params(axis='y', labelsize = label_size)
            sec_ax.tick_params(axis='y', labelsize = label_size)
            
        
        # add a line of the Insertion object onto axc (cax - coronal)
         # plotting PLANNED insertion 
        #axc.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        #axs.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        
        # crop coronal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 0]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 0]) *1e6 + 1000
        
        axc.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axc.axes.set_aspect('equal')
        
        # crop sagittal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 1]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 1]) *1e6 + 1000
        
        axs.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axs.axes.set_aspect('equal')
        
        
        # plot channels on each graph
        
        # subset the data_frame to subject
        subj_frame = ch_data[ch_data['subject'] == subject_ID]
        
        # retrieve the location in XYZ
        locX = subj_frame['chan_loc_x'].values
        locY = subj_frame['chan_loc_y'].values
        locZ = subj_frame['chan_loc_z'].values
        
        # plot channels as circles at x2 the dpi
         # this gives channel coords that are just about separate in the figure!
         # ms=(72./figcor.dpi)*2
        axc.plot(locX * 1e6, locZ * 1e6,  marker='o',
                 ms=0.5, mew=0, 
            color=colour, linestyle="", lw=0)
        
        axs.plot(locY * 1e6, locZ * 1e6, marker='o',
                 ms=0.5, mew=0, 
            color=colour, linestyle="", lw=0)
        
        if col_index != 0:
            # remove the primary y axes
            print('  remove 1st y..')
            axc.get_yaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
        
        # add a scale bar
        if i == 0:
            axc.plot(   [-500, -1500], 
                        [ -7250, -7250 ], 
                        color= colour, linewidth = 2)
        
    
    # adjust spacing
    wspace = 0.3   # the amount of width reserved for blank space between subplots
      # gives the tightest layout without overlap between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots
    
    figcor.subplots_adjust(wspace, hspace)
    figsag.subplots_adjust(wspace, hspace)
    
    # reset the sizes
    #figcor.set_size_inches(3, 2.15)
    #figcor.set_size_inches(3, 2.15)
    figcor.tight_layout()
    figsag.tight_layout()
    
    # save to output
    figcor.savefig( str(Path(OUTPUT, 'B_channels_subj3_hist_coronal.svg')), bbox_inches="tight" )
    figsag.savefig( str(Path(OUTPUT, 'B_channels_subj3_hist_sagittal.svg')), bbox_inches="tight" )
    



def plot_channels_n1(output='figure_histology', remove_axes=True, colour='w'):
    '''Plot one subject, CORONAL & SAGITTAL histology and channels for repeated site
    
    Gets the subjects to plot from the load_channels_data() function in 
    probe_geometry_data module.
    
    Plots all coronal and all sagittal data in one large figure.
    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    from one.api import ONE
    from ibllib.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import ibllib.atlas as atlas
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # connect to ibl server
    one = ONE()
    
    # load channels data for repeated site from local cache
    ch_data = fhd.get_channels_data('-2243_-2000')
    
    # get subject IDs as list
    ids = list(dict.fromkeys(ch_data['subject']))
    
    # get set of ids - one per lab
    
    # get all subjects dict for BWM project
    subjs = one.alyx.rest('subjects', 'list', project='ibl_neuropixel_brainwide_01')
    
    # get all labs for ids
    labs = list()
    for s in ids:
        for m in subjs:
            if m['nickname'] == s:
                labs.append(m['lab'])
    
    # get set of labs - remove dups
    labs_set = set(labs)
    
    # get set of ids where LAST instance of lab from labs_set
    ids_sub = [None] * len(labs_set)
    for j, l in enumerate(labs_set):
        #print(j, l)
        for i, d in enumerate(ids):
            if labs[i] == l:
                ids_sub[j] = d
    
    # just set ids_sub to ids
    ids = ids_sub
    
    # set ids to subset of 3
    #ids = [ids[4], ids[5], ids[6]]
    
    # manually set the IDs
    #ids = ['CSH_ZAD_026', 'KS052', 'SWC_058']
    ids = ['CSH_ZAD_026']
    
    # ids is 3
     # generate a figure with axes - 2 rows, 5 cols (max 10)
      # coronal and sagittal
    nRows = 1
    nCols = 3
    
    # some parameters for fig gen:
    gr_percentile_min=0.5
    gr_percentile_max=99.9
    rd_percentile_min=1
    rd_percentile_max=99.9
    font_size = 6
    label_size = 7
    
    # create figures
    #figcor = plt.figure(figsize=(20, 10), dpi=72)
    #figsag = plt.figure(figsize=(20, 10), dpi=72)
    figcor = plt.figure()
    figsag = plt.figure()
    
    # reset the sizes
    figcor.set_size_inches(3, 2.15)
    figsag.set_size_inches(3, 2.15)
    
    # loop through all ids and generate tilted slice plots with recording sites plotted
    row_index = -1
    col_index = 0
    for i in range(0, len(ids)):
        
        # keep track of row/col
        if(i%nCols == 0):
            row_index=row_index+1
        col_index = i%nCols
        
        print('row ', row_index)
        print('col', col_index)
        
        subject_ID = ids[i]
        sub_index = ch_data['subject']==subject_ID
        lab = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'lab']
        eid = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'eid']
        probe = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'probe']
    
        # Fetch trajectory DICT - 'Histology track'
        traj = one.alyx.rest('trajectories', 'list', session=eid, probe=probe, provenance='Histology track')
    
        # Create insertion OBJECT from trajectory DICT
        ins = Insertion.from_dict(traj[0])
        
        axc = figcor.add_subplot(nRows, nCols, (i+1))
        axs = figsag.add_subplot(nRows, nCols, (i+1))
        
        # get the insertion for id 0 - NYU-12
        print( str(i) + " : " + subject_ID)
        
        # labels for axes
        axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
        
        hist_paths = hist.download_histology_data(subject_ID, lab)
        
        # create the brain atlases from the data
        ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
        ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
        
        #gr_thal_roi = ba_gr.image[150:200, 178:278, 100:200]
        # in Z slices run from ANTERIOR to POSTERIOR (528-150,200)
        gr_thal_roi = ba_gr.image[328:378, 178:278, 100:200] # isolate large slice over thalamus for max pixel value
        
        # CORONAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 1, volume = ba_rd.image)
        
        #gr_tslice_roi = gr_tslice[120:240, 150:300] # isolate large slice over thalamus for max pixel value
        #rd_tslice_roi = rd_tslice[120:240, 150:300]
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and gr_thal_roi max to scale autofl.
          # using rd_tslice min and percentile (99.99 default) to scale CM-DiI
        gr_in = np.interp(gr_tslice, (np.percentile(gr_tslice, gr_percentile_min), 
                                      np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, (np.percentile(rd_tslice, rd_percentile_min), 
                                      np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to imshow()
        axc.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        if remove_axes:
            axc.set_axis_off()
        else:
            sec_ax = axc.secondary_yaxis('right', functions=(
                                lambda x: x * ab[0] + ab[1],
                                lambda y: (y - ab[1]) / ab[0]))
            
            axc.set_xlabel(axis_labels[0], fontsize=font_size)
            axc.set_ylabel(axis_labels[1], fontsize=font_size)
            sec_ax.set_ylabel(axis_labels[2], fontsize=font_size)
            
            axc.tick_params(axis='x', labelrotation = 90)
            
            axc.tick_params(axis='x', labelsize = label_size)
            axc.tick_params(axis='y', labelsize = label_size)
            sec_ax.tick_params(axis='y', labelsize = label_size)
            
        
        # SAGITTAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 0, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 0, volume = ba_rd.image)
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and max to scale the image
           # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in/1.5!
        #gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
        #rd_in = np.interp(rd_tslice, (gr_tslice.min(), gr_tslice.max()/1.5), (0, 255))
        gr_in = np.interp(gr_tslice, 
                          (np.percentile(gr_tslice, gr_percentile_min), 
                              np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, 
                          (np.percentile(rd_tslice, rd_percentile_min), 
                              np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to axs via imshow()
        axs.imshow(Zt, interpolation='none', aspect='auto', 
                    extent=np.r_[width, height], cmap=cmap, 
                    vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        #start = ins.xyz[:, 1] * 1e6
        #end = ins.xyz[:, 2] * 1e6
        #xCoords = np.array([start[0], end[0]])
        if remove_axes:
            axs.set_axis_off()
            
        else:
            sec_ax = axs.secondary_yaxis('right', functions=(
                                lambda x: x * ab[0] + ab[1],
                                lambda y: (y - ab[1]) / ab[0]))
            
            axs.set_xlabel(axis_labels[2], fontsize=font_size)
            axs.set_ylabel(axis_labels[1], fontsize=font_size)
            sec_ax.set_ylabel(axis_labels[0], fontsize=font_size)
            
            axs.tick_params(axis='x', labelrotation = 90)
            
            axs.tick_params(axis='x', labelsize = label_size)
            axs.tick_params(axis='y', labelsize = label_size)
            sec_ax.tick_params(axis='y', labelsize = label_size)
            
        
        # add a line of the Insertion object onto axc (cax - coronal)
         # plotting PLANNED insertion 
        #axc.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        #axs.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        
        # crop coronal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 0]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 0]) *1e6 + 1000
        
        axc.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axc.axes.set_aspect('equal')
        
        # crop sagittal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 1]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 1]) *1e6 + 1000
        
        axs.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axs.axes.set_aspect('equal')
        
        
        # plot channels on each graph
        
        # subset the data_frame to subject
        subj_frame = ch_data[ch_data['subject'] == subject_ID]
        
        # retrieve the location in XYZ
        locX = subj_frame['chan_loc_x'].values
        locY = subj_frame['chan_loc_y'].values
        locZ = subj_frame['chan_loc_z'].values
        
        # plot channels as circles at x2 the dpi
         # this gives channel coords that are just about separate in the figure!
         # ms=(72./figcor.dpi)*2
        axc.plot(locX * 1e6, locZ * 1e6,  marker='o',
                 ms=0.5, mew=0, 
            color=colour, linestyle="", lw=0)
        
        axs.plot(locY * 1e6, locZ * 1e6, marker='o',
                 ms=0.5, mew=0, 
            color=colour, linestyle="", lw=0)
        
        if col_index != 0:
            # remove the primary y axes
            print('  remove 1st y..')
            axc.get_yaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
        
        # add a scale bar
        if i == 0:
            axc.plot(   [-500, -1500], 
                        [ -7250, -7250 ], 
                        color= colour, linewidth = 2)
        
    
    # adjust spacing
    wspace = 0.3   # the amount of width reserved for blank space between subplots
      # gives the tightest layout without overlap between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots
    
    figcor.subplots_adjust(wspace, hspace)
    figsag.subplots_adjust(wspace, hspace)
    
    # reset the sizes
    #figcor.set_size_inches(3, 2.15)
    #figcor.set_size_inches(3, 2.15)
    figcor.tight_layout()
    figsag.tight_layout()
    
    # save to output
    figcor.savefig( str(Path(OUTPUT, 'B_channels_subj1_hist_coronal.svg')), bbox_inches="tight" )
    figsag.savefig( str(Path(OUTPUT, 'B_channels_subj1_hist_sagittal.svg')), bbox_inches="tight" )
    




def plot_channels_per_lab(output='figure_histology'):
    '''Plot one subject per lab, CORONAL & SAGITTAL histology and channels for repeated site
    
    Gets the subjects to plot from the load_channels_data() function in 
    probe_geometry_data module.
    
    Plots all coronal and all sagittal data in one large figure.
    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    from one.api import ONE
    from ibllib.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import ibllib.atlas as atlas
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # connect to ibl server
    one = ONE()
    
    # load channels data for repeated site from local cache
    ch_data = fhd.get_channels_data('-2243_-2000')
    
    # get subject IDs as list
    ids = list(dict.fromkeys(ch_data['subject']))
    
    # get set of ids - one per lab
    
    # get all subjects dict for BWM project
    subjs = one.alyx.rest('subjects', 'list', project='ibl_neuropixel_brainwide_01')
    
    # get all labs for ids
    labs = list()
    for s in ids:
        for m in subjs:
            if m['nickname'] == s:
                labs.append(m['lab'])
    
    # get set of labs - remove dups
    labs_set = set(labs)
    
    # get set of ids where LAST instance of lab from labs_set
    ids_sub = [None] * len(labs_set)
    for j, l in enumerate(labs_set):
        #print(j, l)
        for i, d in enumerate(ids):
            if labs[i] == l:
                ids_sub[j] = d
    
    # just set ids_sub to ids
    ids = ids_sub
    
    # ids is 10
     # generate a figure with axes - 2 rows, 5 cols (max 10)
      # coronal and sagittal
    nRows = 2
    nCols = 5
    
    # some parameters for fig gen:
    gr_percentile_min=0.5
    gr_percentile_max=99.9
    rd_percentile_min=1
    rd_percentile_max=99.9
    font_size = 6
    label_size = 7
    
    # create figures
    figcor = plt.figure(figsize=(20, 10), dpi=72)
    figsag = plt.figure(figsize=(20, 10), dpi=72)
    
    # loop through all ids and generate tilted slice plots with recording sites plotted
    row_index = -1
    col_index = 0
    for i in range(0, len(ids)):
        
        # keep track of row/col
        if(i%nCols == 0):
            row_index=row_index+1
        col_index = i%nCols
        
        print('row ', row_index)
        print('col', col_index)
        
        subject_ID = ids[i]
        sub_index = ch_data['subject']==subject_ID
        lab = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'lab']
        eid = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'eid']
        probe = ch_data.at[ ch_data.index[ sub_index ].tolist()[0] , 'probe']
    
        # Fetch trajectory DICT - 'Histology track'
        traj = one.alyx.rest('trajectories', 'list', session=eid, probe=probe, provenance='Histology track')
    
        # Create insertion OBJECT from trajectory DICT
        ins = Insertion.from_dict(traj[0])
        
        axc = figcor.add_subplot(nRows, nCols, (i+1))
        axs = figsag.add_subplot(nRows, nCols, (i+1))
        
        # get the insertion for id 0 - NYU-12
        print( str(i) + " : " + subject_ID)
        
        # labels for axes
        axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
        
        hist_paths = hist.download_histology_data(subject_ID, lab)
        
        # create the brain atlases from the data
        ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
        ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
        
        #gr_thal_roi = ba_gr.image[150:200, 178:278, 100:200]
        # in Z slices run from ANTERIOR to POSTERIOR (528-150,200)
        gr_thal_roi = ba_gr.image[328:378, 178:278, 100:200] # isolate large slice over thalamus for max pixel value
        
        # CORONAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 1, volume = ba_rd.image)
        
        #gr_tslice_roi = gr_tslice[120:240, 150:300] # isolate large slice over thalamus for max pixel value
        #rd_tslice_roi = rd_tslice[120:240, 150:300]
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and gr_thal_roi max to scale autofl.
          # using rd_tslice min and percentile (99.99 default) to scale CM-DiI
        gr_in = np.interp(gr_tslice, (np.percentile(gr_tslice, gr_percentile_min), 
                                      np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, (np.percentile(rd_tslice, rd_percentile_min), 
                                      np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to imshow()
        axc.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        sec_ax = axc.secondary_yaxis('right', functions=(
                            lambda x: x * ab[0] + ab[1],
                            lambda y: (y - ab[1]) / ab[0]))
        
        axc.set_xlabel(axis_labels[0], fontsize=font_size)
        axc.set_ylabel(axis_labels[1], fontsize=font_size)
        sec_ax.set_ylabel(axis_labels[2], fontsize=font_size)
        
        axc.tick_params(axis='x', labelrotation = 90)
        
        axc.tick_params(axis='x', labelsize = label_size)
        axc.tick_params(axis='y', labelsize = label_size)
        sec_ax.tick_params(axis='y', labelsize = label_size)
        
        # SAGITTAL
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 0, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 0, volume = ba_rd.image)
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and max to scale the image
           # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in/1.5!
        #gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
        #rd_in = np.interp(rd_tslice, (gr_tslice.min(), gr_tslice.max()/1.5), (0, 255))
        gr_in = np.interp(gr_tslice, 
                          (np.percentile(gr_tslice, gr_percentile_min), 
                              np.percentile(gr_thal_roi, gr_percentile_max)), 
                          (0, 255))
        rd_in = np.interp(rd_tslice, 
                          (np.percentile(rd_tslice, rd_percentile_min), 
                              np.percentile(rd_tslice, rd_percentile_max)), 
                          (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                       gr_in.astype(dtype=np.uint8), 
                       np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to axs via imshow()
        axs.imshow(Zt, interpolation='none', aspect='auto', 
                    extent=np.r_[width, height], cmap=cmap, 
                    vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        #start = ins.xyz[:, 1] * 1e6
        #end = ins.xyz[:, 2] * 1e6
        #xCoords = np.array([start[0], end[0]])
        
        sec_ax = axs.secondary_yaxis('right', functions=(
                            lambda x: x * ab[0] + ab[1],
                            lambda y: (y - ab[1]) / ab[0]))
        
        axs.set_xlabel(axis_labels[2], fontsize=font_size)
        axs.set_ylabel(axis_labels[1], fontsize=font_size)
        sec_ax.set_ylabel(axis_labels[0], fontsize=font_size)
        
        axs.tick_params(axis='x', labelrotation = 90)
        
        axs.tick_params(axis='x', labelsize = label_size)
        axs.tick_params(axis='y', labelsize = label_size)
        sec_ax.tick_params(axis='y', labelsize = label_size)
        
        
        # add a line of the Insertion object onto axc (cax - coronal)
         # plotting PLANNED insertion 
        #axc.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        #axs.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
        
        # crop coronal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 0]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 0]) *1e6 + 1000
        
        axc.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axc.axes.set_aspect('equal')
        
        # crop sagittal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 1]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 1]) *1e6 + 1000
        
        axs.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        axs.axes.set_aspect('equal')
        
        
        # plot channels on each graph
        #figs = probe_geom_plots.plot_channels(figs)
        # performing computations locally
        colour='y'
        
        # subset the data_frame to subject
        subj_frame = ch_data[ch_data['subject'] == subject_ID]
        
        # retrieve the location in XYZ
        locX = subj_frame['chan_loc_x'].values
        locY = subj_frame['chan_loc_y'].values
        locZ = subj_frame['chan_loc_z'].values
        
        # plot channels as circles at half the dpi
         # this gives channel coords that are just about separate in the figure!
        axc.plot(locX * 1e6, locZ * 1e6,  marker='o',
                 ms=(72./figcor.dpi)/2, mew=0, 
            color=colour, linestyle="", lw=0)
        
        axs.plot(locY * 1e6, locZ * 1e6, marker='o',
                 ms=(72./figsag.dpi)/2, mew=0, 
            color=colour, linestyle="", lw=0)
        
        if col_index != 0:
            # remove the primary y axes
            print('  remove 1st y..')
            axc.get_yaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
        
    
    # adjust spacing
    wspace = 0.4   # the amount of width reserved for blank space between subplots
      # gives the tightest layout without overlap between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots
    
    figcor.subplots_adjust(wspace, hspace)
    figsag.subplots_adjust(wspace, hspace)
    
    # save to output
    figcor.savefig( str(Path(OUTPUT, 'B_channels_subj_lab_hist_coronal.svg')), bbox_inches="tight" )
    figsag.savefig( str(Path(OUTPUT, 'B_channels_subj_lab_hist_sagittal.svg')), bbox_inches="tight" )
    


if __name__ == "__main__":
    plot_channels_n3() # plotting 3 examples


