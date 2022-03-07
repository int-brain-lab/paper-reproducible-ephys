from one.api import ONE


def download_trajectory_data(x, y,
                             project='ibl_neuropixel_brainwide_01'):
    """Download probe trajectory geometry data for all given probes in a given
    project at the planned insertion coord [x,y] from Alyx.

    Downloads the most up-to-date data from Alyx for all recordings at [x,y]

    Saves this data to a standard location in the file system: one params CACHE_DIR

    Also returns this data as a pandas DataFrame object with following:

    * subject lab eid probe - IDs

    * recording_date

    * planned_[x y z] planned_[theta depth phi]

    * micro_[x y z] micro_[theta depth phi]

    * micro_error_surf micro_error_tip

    * hist_[x y z] hist_[theta depth phi]

    * hist_error_surf hist_error_tip

    * hist_to_micro_error_surf hist_to_micro_error_tip

    * hist_ + micro_ : saggital_angle + _coronal_angle

    * mouse_recording_weight dob

    Parameters
    ----------
    x : int
        x insertion coord in µm.  Eg. repeated site is -2243
    y : int
        y insertion coord in µm. Eg. repeated site is -2000.
    project : str, optional
        Trajectory project to list from. The default is
        'ibl_neuropixel_brainwide_01'.

    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing retrieved data

    """

    from one.api import ONE
    import ibllib.atlas as atlas

    import numpy as np
    import pandas as pd

    # in format -2000_-2243 - added to saved file name
    prefix = str(str(x) + "_" + str(y))

    # connect to ONE
    one = ONE()

    # get the planned trajectory for site [x,y]
    traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=x, y=y, project=project)

    # from this collect all eids, probes, subjects, labs from traj
    eids = [sess['session']['id'] for sess in traj]
    probes = [sess['probe_name'] for sess in traj]
    subj = [sess['session']['subject'] for sess in traj]

    # get planned insertion - can get from any retrieved above in traj
    ins_plan = atlas.Insertion.from_dict(traj[0])

    # create a trajectory object of Planned Repeated Site from this insertion:
    # traj_plan = ins_plan.trajectory

    # new dict to store data from loop:
    # subject lab eid probe - IDs
    # recording_data - date of recording of the probe
    # planned micro hist - xyz theta/depth/phy
    # gives the insertion data xyz brain surface insertion plus angle and length
    # error surf/tip
    # euclidean dist at brain surface or tip between planned and micro/hist
    # or micro and hist
    data = {

        'subject': [],
        'lab': [],
        'eid': [],
        'probe': [],

        'recording_date': [],

        'planned_x': [],
        'planned_y': [],
        'planned_z': [],
        'planned_theta': [],
        'planned_depth': [],
        'planned_phi': [],

        'micro_x': [],
        'micro_y': [],
        'micro_z': [],
        'micro_theta': [],
        'micro_depth': [],
        'micro_phi': [],

        'micro_error_surf': [],
        'micro_error_tip': [],

        'hist_x': [],
        'hist_y': [],
        'hist_z': [],
        'hist_theta': [],
        'hist_depth': [],
        'hist_phi': [],

        'hist_error_surf': [],
        'hist_error_tip': [],

        'hist_to_micro_error_surf': [],
        'hist_to_micro_error_tip': []

    }

    # get new atlas generating histology insertion
    brain_atlas = atlas.AllenAtlas(res_um=25)

    # loop through each eid/probe:
    for eid, probe in zip(eids, probes):

        print(" ")
        print(subj[eids.index(eid)])
        print("==================================================================")
        print(eids.index(eid))
        print(eid)
        print(probe)

        # get the eid/probe as insertion
        insertion = one.alyx.rest('insertions', 'list', session=eid,
                                  name=probe)

        print("insertion ")

        if insertion:

            # check if histology has been traced and loaded
            tracing = insertion[0]['json']

            if tracing is None:
                print("No tracing for this sample - skip")
                continue

            if "xyz_picks" not in tracing:
                print("No tracing for this sample - skip")
                continue

            print("tracing")

            if tracing:

                # For this insertion which has histology tracing, retrieve

                # CURRENT planned trajectory - to get subject and other metadata
                planned = one.alyx.rest('trajectories', 'list', session=eid,
                                        probe=probe, provenance='planned')

                # micro-manipulator trajectory and insertion
                micro_traj = one.alyx.rest('trajectories', 'list', session=eid,
                                           probe=probe, provenance='Micro-manipulator')
                micro_ins = atlas.Insertion.from_dict(micro_traj[0])

                print("micro_traj")

                # get histology trajectory and insertion
                # this retrieves the histology traced track from xyz_picks
                track = np.array(insertion[0]['json']['xyz_picks']) / 1e6
                track_ins = atlas.Insertion.from_track(track, brain_atlas)
                # track_traj = track_ins.trajectory

                print("track_traj")

                # only proceed if micro_traj and track_ins is not None
                if micro_traj is None:
                    print("micro_traj is NONE - skip")
                    print("")
                    continue

                if track_ins is None:
                    print("track_ins is NONE - skip")
                    print("")
                    continue

                data['subject'].append(planned[0]['session']['subject'])
                data['lab'].append(planned[0]['session']['lab'])
                data['eid'].append(eid)
                data['probe'].append(probe)

                data['recording_date'].append(planned[0]['session']['start_time'][:10])

                data['planned_x'].append(planned[0]['x'])
                data['planned_y'].append(planned[0]['y'])
                data['planned_z'].append(planned[0]['z'])
                data['planned_depth'].append(planned[0]['depth'])
                data['planned_theta'].append(planned[0]['theta'])
                data['planned_phi'].append(planned[0]['phi'])

                data['micro_x'].append(micro_traj[0]['x'])
                data['micro_y'].append(micro_traj[0]['y'])
                data['micro_z'].append(micro_traj[0]['z'])
                data['micro_depth'].append(micro_traj[0]['depth'])
                data['micro_theta'].append(micro_traj[0]['theta'])
                data['micro_phi'].append(micro_traj[0]['phi'])

                # compute error from planned
                error = micro_ins.xyz[0, :] - ins_plan.xyz[0, :]
                data['micro_error_surf'].append(np.sqrt(np.sum(error ** 2)) * 1e6)
                error = micro_ins.xyz[1, :] - ins_plan.xyz[1, :]
                data['micro_error_tip'].append(np.sqrt(np.sum(error ** 2)) * 1e6)

                data['hist_x'].append(track_ins.x * 1e6)
                data['hist_y'].append(track_ins.y * 1e6)
                data['hist_z'].append(track_ins.z * 1e6)
                data['hist_depth'].append(track_ins.depth * 1e6)
                data['hist_theta'].append(track_ins.theta)
                data['hist_phi'].append(track_ins.phi)

                # compute error from planned
                error = track_ins.xyz[0, :] - ins_plan.xyz[0, :]
                data['hist_error_surf'].append(np.sqrt(np.sum(error ** 2)) * 1e6)
                error = track_ins.xyz[1, :] - ins_plan.xyz[1, :]
                data['hist_error_tip'].append(np.sqrt(np.sum(error ** 2)) * 1e6)

                # compute error from micro
                error = track_ins.xyz[0, :] - micro_ins.xyz[0, :]
                data['hist_to_micro_error_surf'].append(np.sqrt(np.sum(error ** 2)) * 1e6)
                error = track_ins.xyz[1, :] - micro_ins.xyz[1, :]
                data['hist_to_micro_error_tip'].append(np.sqrt(np.sum(error ** 2)) * 1e6)

    # HISTOLOGY DATA:
    # Using phi and theta calculate angle in SAGITTAL plane (beta)
    x = np.sin(np.array(data['hist_theta']) * np.pi / 180.) * \
        np.sin(np.array(data['hist_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['hist_theta']) * np.pi / 180.)
    # add this data to the list:
    data['hist_saggital_angle'] = np.arctan2(x, y) * 180 / np.pi  # hist_beta

    # Using phi and theta calculate angle in coronal plane (alpha)
    x = np.sin(np.array(data['hist_theta']) * np.pi / 180.) * \
        np.cos(np.array(data['hist_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['hist_theta']) * np.pi / 180.)
    # add this data to the list:
    data['hist_coronal_angle'] = np.arctan2(x, y) * 180 / np.pi  # hist_alpha

    # MICRO MANIPULATOR DATA:
    # Using phi and theta calculate angle in sagittal plane (beta)
    x = np.sin(np.array(data['micro_theta']) * np.pi / 180.) * \
        np.sin(np.array(data['micro_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['micro_theta']) * np.pi / 180.)
    # add this data to the list:
    data['micro_saggital_angle'] = np.arctan2(x, y) * 180 / np.pi  # micro_beta

    # Using phi and theta calculate angle in coronal plane (alpha)
    x = np.sin(np.array(data['micro_theta']) * np.pi / 180.) * \
        np.cos(np.array(data['micro_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['micro_theta']) * np.pi / 180.)
    # add this data to the list:
    data['micro_coronal_angle'] = np.arctan2(x, y) * 180 / np.pi  # micro_alpha

    # Get mouse weights around time of recordings
    # mouse weights from Alyx https://github.com/int-brain-lab/ibllib/issues/50
    rec_wts = []  # empty list to append weights to
    # rec_subjs = [] # also track the subject ID to check its correct!
    # it is correct to commented out :)

    for s, r in zip(data['subject'], data['recording_date']):
        wts = one.alyx.rest('subjects', 'read', s)
        print(s)
        for w in wts['weighings']:
            if w['date_time'][:10] == r:
                print(r)
                rec_wts.append(w['weight'])
                # rec_subjs.append(s)
                break  # only add one weight per subject

    data['mouse_recording_weight'] = rec_wts
    # data['rec_subj'] = rec_subjs

    # get dobs
    dobs = []
    for s in data['subject']:
        subject_list = one.alyx.rest('subjects', 'list', nickname=s)
        dobs.append(subject_list[0]['birth_date'])

    data['dob'] = dobs

    # compute days alive at recording from dob and recording_date
    # age_days = []
    # for d in range(len(data)):
    # age_days.append((date( int(data['recording_date'][d][:4]), int(data['recording_date'][d][5:7]), int(data['recording_date'][d][8:10]) ) - date( int(data['dob'][d][:4]), int(data['dob'][d][5:7]), int(data['dob'][d][8:10]) ) ).days)

    # convert data to a Pandas DataFrame:
    data_frame = pd.DataFrame.from_dict(data)

    save_trajectory_data(data_frame, prefix, project)

    return data_frame
