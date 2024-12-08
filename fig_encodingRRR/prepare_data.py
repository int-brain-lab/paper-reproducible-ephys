from utils.download_data import load_data_from_eid, filter_trials_func, filter_neurons_func

import numpy as np
from tqdm import tqdm
import os, glob, shutil

from one.api import ONE
from iblatlas.atlas import AllenAtlas


cache_folder = "./data/cache/"
data_folder = "./data/rep_ephys_22032024/downloaded/"

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', 
          cache_dir=cache_folder)


ba = AllenAtlas()
br = ba.regions



eids = ['a4a74102-2af5-45dc-9e41-ef7f5aed88be', 'd57df551-6dcb-4242-9c72-b806cff5613a', '56b57c38-2699-4091-90a8-aba35103155e', '72cb5550-43b4-4ef0-add5-e4adfdfb5e02', '0c828385-6dd6-4842-a702-c5075f5f5e81', '746d1902-fa59-4cab-b0aa-013be36060d5', 'dac3a4c1-b666-4de0-87e8-8c514483cacf', 'caa5dddc-9290-4e27-9f5e-575ba3598614', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'f115196e-8dfe-4d2a-8af3-8206d93c1729', '0a018f12-ee06-4b11-97aa-bbbff5448e9f', 'd2832a38-27f6-452d-91d6-af72d794136c', '6f09ba7e-e3ce-44b0-932b-c003fb44fb89', 'b196a2ad-511b-4e90-ac99-b5a29ad25c22', '73918ae1-e4fd-4c18-b132-00cb555b1ad2', 'f312aaec-3b6f-44b3-86b4-3a0c119c0438', 'dda5fc59-f09a-4256-9fb5-66c67667a466', 'ebce500b-c530-47de-8cb1-963c552703ea', 'ee40aece-cffd-4edb-a4b6-155f158c666a', '614e1937-4b24-4ad3-9055-c8253d089919', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '61e11a11-ab65-48fb-ae08-3cb80662e5d6', 'ecb5520d-1358-434c-95ec-93687ecd1396', '54238fd6-d2d0-4408-b1a9-d19d24fd29ce', 'e45481fa-be22-4365-972c-e7404ed8ab5a', 'e535fb62-e245-4a48-b119-88ce62a6fe67', 'a66f1593-dafd-4982-9b66-f9554b6c86b5', '51e53aff-1d5d-4182-a684-aba783d50ae5', '0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63', '111c1762-7908-47e0-9f40-2f2ee55b6505', '0802ced5-33a3-405e-8336-b65ebc5cb07c', 'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0', '3f859b5c-e73a-4044-b49e-34bb81e96715', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '8928f98a-b411-497e-aa4b-aa752434686d', '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca', '7cb81727-2097-4b52-b480-c89867b5b34c', '30c4e2ab-dffc-499d-aae4-e51d6b3218c2', '71e55bfe-5a3a-4cba-bdc7-f085140d798e', 'd04feec7-d0b7-4f35-af89-0232dd975bf0', '824cf03d-4012-4ab1-b499-c83a92c5589e', '5ae68c54-2897-4d3a-8120-426150704385', '41872d7f-75cb-4445-bb1a-132b354c44f0', 'c7bf2d49-4937-4597-b307-9f39cb1c7b16', 'e2b845a1-e313-4a08-bc61-a5f662ed295e', '4a45c8ba-db6f-4f11-9403-56e06a33dfa4', '7af49c00-63dd-4fed-b2e0-1b3bd945b20b', '15763234-d21e-491f-a01b-1238eb96d389', '781b35fd-e1f0-4d14-b2bb-95b7263082bb', '754b74d5-7a06-4004-ae0c-72a10b6ed2e6', '4b00df29-3769-43be-bb40-128b1cba6d35', 'f140a2ec-fd49-4814-994a-fe3476f14e66', '687017d4-c9fc-458f-a7d5-0979fe1a7470', '862ade13-53cd-4221-a3fa-dda8643641f2', '3638d102-e8b6-4230-8742-e548cd87a949', 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5', '6899a67d-2e53-4215-a52a-c7021b5da5d4', '15b69921-d471-4ded-8814-2adad954bcd8', '2bdf206a-820f-402f-920a-9e86cd5388a4', 'd9f0c293-df4c-410a-846d-842e47c6b502', '88224abb-5746-431f-9c17-17d7ef806e6a', 'aad23144-0e52-4eac-80c5-c4ee2decb198', 
        'd0ea3148-948d-4817-94f8-dcaf2342bbbe', 'b22f694e-4a34-4142-ab9d-2556c3487086', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', 
        '642c97ea-fe89-4ec9-8629-5e492ea4019d', 'ff96bfe1-d925-4553-94b5-bf8297adf259', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', 
        'c4432264-e1ae-446f-8a07-6280abade813', 'dc962048-89bb-4e6a-96a9-b062a2be1426', '38d95489-2e82-412a-8c1a-c5377b5f1555', 
        '0841d188-8ef2-4f20-9828-76a94d5343a4', '91bac580-76ed-41ab-ac07-89051f8d7f6e', '9b528ad0-4599-4a55-9148-96cc1d93fb24', 
        '6c6b0d06-6039-4525-a74b-58cfaa1d3a60', '7f6b86f9-879a-4ea2-8531-294a221af5d0', '1b715600-0cbc-442c-bd00-5b0ac2865de1', 
        'd23a44ef-1402-4ed7-97f5-47e9a7a504d9', '8a3a0197-b40a-449f-be55-c00b23253bbf', '3e6a97d3-3991-49e2-b346-6948cb4580fb']
checked_eids = []
for eid in tqdm(eids):

    neural_fname = f"neural_data_{eid}.npz"
    beh_fname = f"beh_data_{eid}.npz"
    if os.path.isfile(os.path.join(data_folder, neural_fname)) and os.path.isfile(os.path.join(data_folder, beh_fname)):
        checked_eids.append(eid)
        continue

    if eid in checked_eids:
        print("already checked this eid, continue")
        continue
    checked_eids.append(eid)


    data_ret = load_data_from_eid(eid, one, ba,
                                    lambda _: filter_trials_func(_,
                                                                remove_timeextreme_event=[True, 0.8],
                                                                remove_no_choice=True),
                                    lambda _: filter_neurons_func(_,
                                                                remove_frextreme=(True, .1, 50.),
                                                                only_goodneuron=(False),
                                                                only_area=(False)),
                                    min_neurons=5, 
                                    spsdt=0.01, Twindow=1.2, t_bf_stimOn=0.2, # sec
                                    load_motion_energy=True, load_wheel_velocity=True,
                                    load_tongue=True)
    if data_ret is None:
        continue


    np.savez(os.path.join(data_folder, neural_fname),
                spike_count_matrix=data_ret['spike_count_matrix'],
                clusters_g=data_ret['clusters_g'],
                eid=eid,
                )
    

    np.savez(os.path.join(data_folder, beh_fname),
                behavior=data_ret['behavior'],
                timeline=data_ret['timeline'],
                eid=eid,
                wheel_vel=data_ret['wheel_vel'],
                whisker_motion=data_ret['whisker_motion'],
                licks=data_ret["licks"],
                )


    for fname in glob.glob(os.path.join(cache_folder, "*")):
        if os.path.isdir(fname):
            shutil.rmtree(fname)
