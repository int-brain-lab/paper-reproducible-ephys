
def global_params(which_areas="all_new", var_types="all_new", inc_type='high_fr'):
    gp = dict(wa=which_areas, vt=var_types, it=inc_type)
    gp['wa'] = which_areas
    if which_areas == "repsite":
        gp['data_id'] = "rep_ephys_22032024"
        gp['al_include'] = ['VISa', 'VISam', 'CA1','DG','LP','PO']
    else: 
        assert False, "invalid which_area"

    if inc_type == 'good_unit_Xstandardized':
        gp['X_inc'] = {"min_trials": 100, "remove_block5": True, "standardize_X": True}
        gp['y_inc'] = {'smooth_w':2., 'min_mfr':.5, 'max_sp': 0.5, "min_neurons":5,
                "transform_mfr": None, "standardize_y": True, "unit_label_min": 1.}
    else: 
        assert False, "invalid inc_type"
    
    if var_types == "clean":
        gp["vl"] = ['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker_max", "lick",]
        gp['v2i'] = {'block':[0,], 'side':[1], 'contrast_level': [2], 'stimulus':[1,2], 'choice': [3], 
                     'reward': [4], 'outcome': [4], 
                    "wheel":[5], "whisker_max":[6], "lick":[7], 
                    'all': list(range(8))}

        gp['tl'] = ['block','contrast_level','choice','outcome'][::-1]
        gp['bl'] = ['wheel','whisker_max','lick'] 
    else: 
        assert False, "invalid var_types"

    if (which_areas == "repsite") and (var_types == "clean") and (inc_type == "good_unit_Xstandardized"):
        gp['RRRGDglobal_p'] = dict(n_comp_list=list(range(3,6)), l2_list=[75,150,300], lr=1.)  # model 4
    else: 
        assert False, "invalid var_types"
    
    return gp

