#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:30:04 2021

Python module for downloading histology data on probes and channels to store to
cache.  Using the get_ methods here collects local cache and performs a manual 
exclusion of some subjects for which histology is poor or there are other
errors.

@author: sjwest
"""



def print_path():
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    print(path)
    



def exclusions():
    '''
    returns list of subjects to be excluded for histology query.

    Returns
    -------
    List of subject IDs [str] to be excluded from datasets.

    '''
    return[
     'CSH_ZAD_001',   # very poor anatomy - dissection damage
     'CSHL051',       # very poor anatomy - dissection damage
     'NYU-47',        # no histology ???
     'UCLA011',       # currently repeated site WAYYY OUT!
     'KS051',         # not been imaged..?
     'KS055',         # error on surface is WAYYY OUT!
     'NYU-27'         # error on surface is WAYYY OUT!
     ]
    



def download_probe_data(x=-2243, y=-2000):
    '''
    Download probe data from IBL Alyx and store to local cache.

    Returns
    -------
    None.

    '''
    from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    
    # ldownoad data and save to local cache
    probe_geom_data.download_trajectory_data(x=x, y=y)
    



def get_probe_data(coord = '-2243_-2000'):
    '''
    Get probe data and perform manual query for histology.

    Returns
    -------
    None.

    '''
    
    from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    
    # load data from local cache
    probe_data = probe_geom_data.load_trajectory_data(coord)
    
    # histology query - manual exclusions
    excl = exclusions()
    for e in excl:
        index = probe_data[ probe_data['subject'] == e ].index
        probe_data.drop(index , inplace=True)
    
    return probe_data
    



def get_channels_data(coord = '-2243_-2000'):
    '''
    Loads channels data from local cache, performs manual histology exclusions
    then returns the data frame.

    Returns
    -------
    Dataframe containing the channels data.

    '''
    from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    
    # load channels data from local cache
    ch_data = probe_geom_data.load_channels_data(coord)
    
    # histology query - manual exclusions
    excl = exclusions()
    for e in excl:
        index = ch_data[ ch_data['subject'] == e ].index
        ch_data.drop(index , inplace=True)
    
    return ch_data
    



def download_channels_data(x=-2243, y=-2000):
    '''
    Download channels data from IBL Alyx and store to local cache.

    Returns
    -------
    None.

    '''
    from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    
    # download channels data for repeated site and save to local cache
    probe_geom_data.download_channels_data(x=x, y=y)
    


#if __name__ == "__main__":
    #get_probe_data()
    #get_channels_data()
    # DO NOTTHING!


