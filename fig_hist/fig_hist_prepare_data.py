#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  1 09:11:26 2020

Probe Geometry Data Collection

@author: stevenwest
'''

from one.api import ONE
from iblatlas.atlas import Insertion, AllenAtlas
from iblutil.numerical import ismember
from brainbox.io.one import load_channel_locations
from reproducible_ephys_functions import save_data_path, save_dataset_info, get_insertions
from fig_hist.fig_hist_load_data import load_dataframe

import numpy as np
import pandas as pd


def prepare_data(insertions, one=None, brain_atlas=None, recompute=False):
    """Download channels geometry data for all given probes in a given project
    at the planned insertion coord [x,y] from Alyx.

    Downloads the most up-to-date data from Alyx for all recordings planned at
    [x,y], including their channel positions, and the orthogonal coords of
    each channel from the planned trajectory.

    Saves this data to a standard location in the file system.

    Also returns this data as a pandas DataFrame object with following cols:

    * subject, eid, probe - the subject, eid and probe IDs
    * chan_loc - xyz coords of all channels
    * planned_orth_proj - xyz coord of orthogonal line from chan_loc onto planned proj
    * dist - the euclidean distance between chan_loc xyz and planned_orth_proj xyz

    """

    if not recompute:
        data_probe = load_dataframe(df_name='traj', exists_only=True)
        data_chns = load_dataframe(df_name='chns', exists_only=True)
        if data_probe and data_chns:
            df_probe = load_dataframe(df_name='traj')
            pids = np.array([p['probe_insertion'] for p in insertions])
            isin, _ = ismember(pids, df_probe['pid'].unique())
            if np.all(isin):
                print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
                df_chns = load_dataframe(df_name='chns')
                return df_probe, df_chns

    brain_atlas = brain_atlas or AllenAtlas()
    brain_atlas.compute_surface()
    # Get the planned trajectory metadata
    ins_plan = Insertion.from_dict(insertions[0], brain_atlas=brain_atlas)

    all_df_chns = []
    all_df_traj = []

    for iIns, ins in enumerate(insertions):
        print(f'processing {iIns + 1}/{len(insertions)}')

        eid = ins['session']['id']
        lab = ins['session']['lab']
        subject = ins['session']['subject']
        date = ins['session']['start_time'][:10]
        pid = ins['probe_insertion']
        probe = ins['probe_name']

        # Trajectory data frame
        try:
            traj_hist = one.alyx.rest('trajectories', 'list', probe_insertion=pid, provenance='Histology track')[0]
            ins_hist = Insertion.from_dict(traj_hist, brain_atlas=brain_atlas)
            traj_micro = one.alyx.rest('trajectories', 'list', probe_insertion=pid, provenance='Micro-manipulator')[0]
            ins_micro = Insertion.from_dict(traj_micro, brain_atlas=brain_atlas)
        except Exception as err:
            print(f'No trajectories for insertion: {pid} - {err}')
            continue

        data_traj = {}

        # Add in all info
        data_traj['eid'] = eid
        data_traj['pid'] = pid
        data_traj['subject'] = subject
        data_traj['probe'] = probe
        data_traj['date'] = date
        data_traj['lab'] = lab

        data_traj['planned_x'] = ins['x']
        data_traj['planned_y'] = ins['y']
        data_traj['planned_z'] = ins['z']
        data_traj['planned_depth'] = ins['depth']
        data_traj['planned_theta'] = ins['theta']
        data_traj['planned_phi'] = ins['phi']

        data_traj['micro_x'] = traj_micro['x']
        data_traj['micro_y'] = traj_micro['y']
        data_traj['micro_z'] = traj_micro['z']
        data_traj['micro_depth'] = traj_micro['depth']
        data_traj['micro_theta'] = traj_micro['theta']
        data_traj['micro_phi'] = traj_micro['phi']

        data_traj['hist_x'] = traj_hist['x']
        data_traj['hist_y'] = traj_hist['y']
        data_traj['hist_z'] = traj_hist['z']
        data_traj['hist_depth'] = traj_hist['depth']
        data_traj['hist_theta'] = traj_hist['theta']
        data_traj['hist_phi'] = traj_hist['phi']

        # compute micro error from planned using ml, ap and dv
        error = ins_micro.xyz[0, :] - ins_plan.xyz[0, :]
        data_traj['micro_error_surf_xyz'] = np.sqrt(np.sum(error ** 2)) * 1e6
        error = ins_micro.xyz[1, :] - ins_plan.xyz[1, :]
        data_traj['micro_error_tip_xyz'] = np.sqrt(np.sum(error ** 2)) * 1e6

        # compute micro error from planned using just ml and ap
        error = ins_micro.xyz[0, :2] - ins_plan.xyz[0, :2]
        data_traj['micro_error_surf_xy'] = np.sqrt(np.sum(error ** 2)) * 1e6
        error = ins_micro.xyz[1, :2] - ins_plan.xyz[1, :2]
        data_traj['micro_error_tip_xy'] = np.sqrt(np.sum(error ** 2)) * 1e6

        # compute hist error from planned using ml, ap and dv
        error = ins_hist.xyz[0, :] - ins_plan.xyz[0, :]
        data_traj['hist_error_surf_xyz'] = np.sqrt(np.sum(error ** 2)) * 1e6
        error = ins_hist.xyz[1, :] - ins_plan.xyz[1, :]
        data_traj['hist_error_tip_xyz'] = np.sqrt(np.sum(error ** 2)) * 1e6

        # compute hist error from planned using just ml and ap
        error = ins_hist.xyz[0, :2] - ins_plan.xyz[0, :2]
        data_traj['hist_error_surf_xy'] = np.sqrt(np.sum(error ** 2)) * 1e6
        error = ins_hist.xyz[1, :2] - ins_plan.xyz[1, :2]
        data_traj['hist_error_tip_xy'] = np.sqrt(np.sum(error ** 2)) * 1e6

        # compute hist error from micro using ml, ap and dv
        error = ins_hist.xyz[0, :] - ins_micro.xyz[0, :]
        data_traj['hist_to_micro_error_surf_xyz'] = np.sqrt(np.sum(error ** 2)) * 1e6
        error = ins_hist.xyz[1, :] - ins_micro.xyz[1, :]
        data_traj['hist_to_micro_error_tip_xyz'] = np.sqrt(np.sum(error ** 2)) * 1e6

        # compute hist error from micro using just ml and ap
        error = ins_hist.xyz[0, :2] - ins_micro.xyz[0, :2]
        data_traj['hist_to_micro_error_surf_xy'] = np.sqrt(np.sum(error ** 2)) * 1e6
        error = ins_hist.xyz[1, :2] - ins_micro.xyz[1, :2]
        data_traj['hist_to_micro_error_tip_xy'] = np.sqrt(np.sum(error ** 2)) * 1e6

        # compute some angles

        # compute the difference in entry coord between ins_plan and ins_hist and use this to move the ins_hist TIP coord
        # ins_plan.xyz[0, :] surface and ins_plan.xyz[1, :] tip
        ins_diff = ins_plan.xyz[0, :] - ins_hist.xyz[0, :]  # get difference between surface insertions
        # ins_hist.xyz[0,:] + ins_diff # GIVES ins_plan.xyz[0,:] !!
        # i.e it MOVES the surface coord of ins_hist to ins_plan surface coord!
        hist_tip_norm = ins_hist.xyz[1, :] + ins_diff  # gives tip coord, with surface moved to planned ins coord
        # ins_plan.xyz[1,:] ins_hist[1,:] - coords are ML, AP, DV
        # want to IGNORE DV - look at direction between coords in ML, AP

        # calculate the ANGLE between ins_plan TIP -> SURFACE -> hist_tip_norm
        # ins_plan.xyz[1, :] -> ins_plan.xyz[0, :] -> hist_tip_norm
        # from: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates # noqa
        ba = ins_plan.xyz[1, :] - ins_plan.xyz[0, :]
        bc = hist_tip_norm - ins_plan.xyz[0, :]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        angle = np.arccos(cosine_angle)  # returns angle in RADIANS
        angle_deg = np.degrees(angle)  # can use degrees if desired
        data_traj['angle'] = angle_deg

        # want to PLOT the MAGNITUDE of this angle!

        # NEXT compute the DIRECTION of the angle from planned to hist get this by first computing the normal vector from
        # hist_tip_norm to the planned trajectory

        # project hist_tip_norm to the PLANNED trajectory:
        hist_tip_norm_vector = ins_plan.trajectory.project(hist_tip_norm)

        # this gives the coord on ins_plan which makes a right angle with hist_tip_norm - so together these coords give
        # the NORMAL VECTOR to ins_plan

        # from the NORMAL VECTOR can compute the ML and AP distances by simply subtracting the two values in the vector
        ml_dist = hist_tip_norm[0] - hist_tip_norm_vector[0]
        ap_dist = hist_tip_norm[1] - hist_tip_norm_vector[1]

        # and thus can compute the ABSOLUTE RATIO that the ANGLE MAGNITUDE is shared between ml and ap
        ml_ratio = np.abs(ml_dist) / (np.abs(ml_dist) + np.abs(ap_dist))
        ap_ratio = np.abs(ap_dist) / (np.abs(ml_dist) + np.abs(ap_dist))

        # combining this ratio with the SIGN of ML_dist/AP_dist can be used to compute the coord
        # of the ANGLE MAGNITUDE from 0,0 as the planned trajectory

        # using pythagoras - compute the distance of the hypotenuse if using ML_ratio and AP_ratio as the right angle lengths
        hyp = np.sqrt((ml_ratio * ml_ratio) + (ap_ratio * ap_ratio))

        # use this to calculate the proportions of each ratio
        angle_ml = (angle_deg / hyp) * ml_ratio
        angle_ap = (angle_deg / hyp) * ap_ratio

        # confirm this works by checking the total length is correct with pythagoras
        # np.sqrt( (angle_ML*angle_ML) + (angle_AP*angle_AP))

        # finally, flip the sign depending on if ML/AP_dist is POSITIVE or NEGATIVE
        if ml_dist < 0:
            angle_ml = -(angle_ml)

        if ap_dist < 0:
            angle_ap = -(angle_ap)

        data_traj['angle_ml'] = angle_ml
        data_traj['angle_ap'] = angle_ap

        # Get subject weight at time of recording and date of birth
        # weight = one.alyx.rest('weighings', 'list', nickname=subject, django=f'date_time__icontains,{date}')[0]
        # data_traj['mouse_recording_weight'] = weight['weight']
        subj = one.alyx.rest('subjects', 'list', nickname=subject)[0]
        data_traj['dob'] = subj['birth_date']

        all_df_traj.append(pd.DataFrame.from_dict([data_traj]))

        # Channel data frame
        try:
            channels = load_channel_locations(eid, probe=probe, one=one, brain_atlas=brain_atlas)[probe]
            ch_loc = np.c_[channels['x'], channels['y'], channels['z']]
            proj = ins_plan.trajectory.project(ch_loc)
            # calculate the distance between proj and ch_loc:
            dist = np.linalg.norm(ch_loc - proj)
        except Exception as err:
            print(f'No channels for insertion: {pid} - {err}')
            continue

        # Channel data
        # Dict to store data:
        # chan_loc - xyz coord of channels
        # planned_orth_proj - xyz coord of orthogonal line from chan_loc to planned proj
        # dist - the 3D distance between chan_loc xyz and planned_orth_proj xyz
        data_chns = {}
        data_chns['chan_loc_x'] = ch_loc[:, 0]
        data_chns['chan_loc_y'] = ch_loc[:, 1]
        data_chns['chan_loc_z'] = ch_loc[:, 2]
        data_chns['planned_orth_proj_x'] = proj[:, 0]
        data_chns['planned_orth_proj_y'] = proj[:, 1]
        data_chns['planned_orth_proj_z'] = proj[:, 2]
        data_chns['dist'] = dist

        df_chns = pd.DataFrame.from_dict(data_chns)
        df_chns['eid'] = eid
        df_chns['pid'] = pid
        df_chns['subject'] = subject
        df_chns['probe'] = probe
        df_chns['date'] = date
        df_chns['lab'] = lab
        all_df_chns.append(df_chns)

    # Concat dataframes from all insertions
    save_path = save_data_path(figure='fig_hist')
    print(f'Saving data to {save_path}')
    concat_df_chns = pd.concat(all_df_chns, ignore_index=True)
    concat_df_traj = pd.concat(all_df_traj, ignore_index=True)
    concat_df_traj = concat_df_traj.drop_duplicates(subset='subject')
    concat_df_chns.to_csv(save_path.joinpath('fig_hist_dataframe_chns.csv'))
    concat_df_traj.to_csv(save_path.joinpath('fig_hist_dataframe_traj.csv'))

    return concat_df_chns, concat_df_traj


if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True
    insertions = get_insertions(level=-1, one=one, freeze='freeze_2024_03')
    all_df_chns, all_df_traj = prepare_data(insertions, one=one)
    save_dataset_info(one, figure='fig_hist')
