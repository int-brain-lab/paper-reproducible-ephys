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


def plot_hist_figure(output_dir='figure_histology', 
                     input_dir='figure_histology'):
    '''
    

    Returns
    -------
    None.

    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import svgutils.compose as sc # compose figure with svgutils
    # I have installed this package to iblenv LOCALLY
    
    import figure_hist_plot_channels_all_subjs as cas
    from scratch_scripts.SW import figure_hist_plot_probe_trajs_ccf as ptc
    from scratch_scripts.SW import figure_hist_plot_probe_surf_coord_MLxAP as psc
    from scratch_scripts.SW import figure_hist_plot_probe_angle_MLxAP as ppa

    print('')
    print('PLOT HISTOLOGY FIGURE')
    print('=====================')
    print('')
    # output DIR
    OUTPUT = Path(output_dir).resolve()
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT) # generate output DIR for storing plots
    
    # download probe and channels data from IBL ONE
    #fhd.download_channels_data()
    #fhd.download_probe_data() # both will collect data from repeated site by default
    
    print(' run get_probe_data()') #get probe_data histology query
    probe_data = fhd.get_probe_data()
    
    # generate all SVG components for panels
    #print(' run plot_channels_n3()')
    #cas.plot_channels_n3(output = input_dir)
    print(' run plot_channels_n3()')
    cas.plot_channels_n3(output = input_dir)
    
    print(' run plot_trajs()')
    ptc.plot_trajs(probe_data, output = input_dir)
    
    print(' run plot_probe_surf_coord_micro_panel()')
    psc.plot_probe_surf_coord_micro_panel(output = input_dir)
    
    print(' run plot_probe_surf_coord_histology_panel()')
    psc.plot_probe_surf_coord_histology_panel(output = input_dir)
    
    print(' run plot_probe_angle_histology_panel()')
    ppa.plot_probe_angle_histology_panel(output = input_dir)
    
    font_size = 3
    # compose figure with svgutils
    fig = sc.Figure("200mm", "190mm", 
        
        sc.Panel(
            
            sc.Text("a", 1, 2.5, size=font_size, weight='bold'),
                    # was 'A_2P-registration.svg'
            sc.SVG(input_dir + os.path.sep + 'A_histology_pipeline_analysis.svg'
                     ).scale(0.95
                     ).move(3,0),
        
            ),
        
        sc.Panel(
            sc.Text("b", 0, 2.5, size=font_size, weight='bold'),
            
            sc.SVG(input_dir + os.path.sep + 'B_channels_subj3_hist_coronal.svg'
                     ).scale(0.3).move(2,0)
        
        ).move(90, 0),
        
        sc.Panel(
                sc.Text("c", 0, 2.5, size=font_size, weight='bold'),
                
                sc.SVG(input_dir + os.path.sep + 'C_probe_trajs_ccf_coronal.svg'
                     ).scale(0.34),
                
                sc.SVG(input_dir + os.path.sep + 'C_probe_trajs_ccf_sagittal.svg'
                     ).scale(0.34).move(22, 0),
                
                ).move(153, 0),
        
        
        #sc.Panel(
        #        
        #    sc.Text("c", 0, 2.5, size=font_size, weight='bold'),
        #    
        #        sc.Image(50, 50, 
        #                 input_dir + os.path.sep + 'C_probe_traj_ccf_3D.png'
        #                 ).scale(0.8).move(2,5)
        #        
        #        ).move(152, 0),
        
        
        sc.Panel(
                sc.Text("d", 1, 2.5, size=font_size, weight='bold'),
                
                sc.SVG(input_dir + os.path.sep + 'surf_coord_micro_panel.svg'
                     ).scale(1.0),
                     
            ).move(0, 47),
        
        
        sc.Panel(
                sc.Text("e", 0, 2.5, size=font_size, weight='bold'),
                
                sc.SVG(input_dir + os.path.sep + 'surf_coord_histology_panel.svg'
                     ).scale(1.0),
                     
            ).move(67, 47),
        
        
        sc.Panel(
                sc.Text("f", 0, 2.5, size=font_size, weight='bold'),
                
                sc.SVG(input_dir + os.path.sep + 'angle_histology_panel.svg'
                     ).scale(1.0),
                     
            ).move(133, 47),
        
        #sc.Grid(20,20)
        
          )
    
    fig.save( Path(output_dir, "figure_histology.svg") )
    


if __name__ == "__main__":
    plot_hist_figure() # generate the whole figure


