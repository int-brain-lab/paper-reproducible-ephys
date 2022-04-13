import logging

from pathlib import Path
from one.api import ONE

one = ONE()

spike_sorter_name = 'pykilosort'
# 0 ce397420-3cd2-4a55-8fd1-5e28321981f4 56b57c38-2699-4091-90a8-aba35103155e /mnt/s0/Data/mrsicflogellab/Subjects/SWC_054/2020-10-05/001/alf/probe01
# 1 e31b4e39-e350-47a9-aca4-72496d99ff2a 746d1902-fa59-4cab-b0aa-013be36060d5 /mnt/s0/Data/mainenlab/Subjects/ZFM-01592/2020-10-20/001/alf/probe00
# 2 f8d0ecdc-b7bd-44cc-b887-3d544e24e561 7b26ce84-07f9-43d1-957f-bc72aeb730a3 /mnt/s0/Data/angelakilab/Subjects/NYU-27/2020-09-28/001/alf/probe00
# 3 6fc4d73c-2071-43ec-a756-c6c6d8322c8b dac3a4c1-b666-4de0-87e8-8c514483cacf /mnt/s0/Data/hoferlab/Subjects/SWC_060/2020-11-24/001/alf/probe01
# 4 c17772a9-21b5-49df-ab31-3017addea12e 6f09ba7e-e3ce-44b0-932b-c003fb44fb89 /mnt/s0/Data/hoferlab/Subjects/SWC_043/2020-09-16/002/alf/probe01
# 5 0851db85-2889-4070-ac18-a40e8ebd96ba 73918ae1-e4fd-4c18-b132-00cb555b1ad2 /mnt/s0/Data/wittenlab/Subjects/ibl_witten_27/2021-01-21/001/alf/probe01
# 6 eeb27b45-5b85-4e5c-b6ff-f639ca5687de f312aaec-3b6f-44b3-86b4-3a0c119c0438 /mnt/s0/Data/churchlandlab/Subjects/CSHL058/2020-07-07/001/alf/probe00
# 7 69f42a9c-095d-4a25-bca8-61a9869871d3 dda5fc59-f09a-4256-9fb5-66c67667a466 /mnt/s0/Data/churchlandlab/Subjects/CSHL059/2020-03-06/001/alf/probe00
# 8 f03b61b4-6b13-479d-940f-d1608eb275cc ee40aece-cffd-4edb-a4b6-155f158c666a /mnt/s0/Data/mainenlab/Subjects/ZM_2241/2020-01-30/001/alf/probe00
# 9 f2ee886d-5b9c-4d06-a9be-ee7ae8381114 ecb5520d-1358-434c-95ec-93687ecd1396 /mnt/s0/Data/churchlandlab/Subjects/CSHL051/2020-02-05/001/alf/probe00
# 10 f26a6ab1-7e37-4f8d-bb50-295c056e1062 54238fd6-d2d0-4408-b1a9-d19d24fd29ce /mnt/s0/Data/danlab/Subjects/DY_018/2020-10-15/001/alf/probe00
# 11 c4f6665f-8be5-476b-a6e8-d81eeae9279d e535fb62-e245-4a48-b119-88ce62a6fe67 /mnt/s0/Data/danlab/Subjects/DY_013/2020-03-12/001/alf/probe00
# 12 9117969a-3f0d-478b-ad75-98263e3bfacf b03fbc44-3d8e-4a6c-8a50-5ea3498568e0 /mnt/s0/Data/danlab/Subjects/DY_010/2020-01-27/001/alf/probe00
# 13 febb430e-2d50-4f83-87a0-b5ffbb9a4943 db4df448-e449-4a6f-a0e7-288711e7a75a /mnt/s0/Data/danlab/Subjects/DY_009/2020-02-27/001/alf/probe00
# 14 8413c5c6-b42b-4ec6-b751-881a54413628 064a7252-8e10-4ad6-b3fd-7a88a2db5463 /mnt/s0/Data/zadorlab/Subjects/CSH_ZAD_029/2020-09-09/001/alf/probe00
# 15 8b7c808f-763b-44c8-b273-63c6afbc6aae 41872d7f-75cb-4445-bb1a-132b354c44f0 /mnt/s0/Data/mrsicflogellab/Subjects/SWC_038/2020-07-29/001/alf/probe01
# 16 f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c dfd8e7df-dc51-4589-b6ca-7baccfeb94b4 /mnt/s0/Data/churchlandlab/Subjects/CSHL045/2020-02-25/002/alf/probe00
# 17 63517fd4-ece1-49eb-9259-371dc30b1dd6 4a45c8ba-db6f-4f11-9403-56e06a33dfa4 /mnt/s0/Data/danlab/Subjects/DY_020/2020-09-29/001/alf/probe00
# 18 8d59da25-3a9c-44be-8b1a-e27cdd39ca34 4b00df29-3769-43be-bb40-128b1cba6d35 /mnt/s0/Data/churchlandlab/Subjects/CSHL052/2020-02-21/001/alf/probe00
# 19 19baa84c-22a5-4589-9cbd-c23f111c054c 862ade13-53cd-4221-a3fa-dda8643641f2 /mnt/s0/Data/hoferlab/Subjects/SWC_042/2020-07-15/001/alf/probe01
# 20 143dd7cf-6a47-47a1-906d-927ad7fe9117 3638d102-e8b6-4230-8742-e548cd87a949 /mnt/s0/Data/mrsicflogellab/Subjects/SWC_058/2020-12-07/001/alf/probe01
# 21 84bb830f-b9ff-4e6b-9296-f458fb41d160 c7248e09-8c0d-40f2-9eb4-700a8973d8c8 /mnt/s0/Data/mainenlab/Subjects/ZM_3001/2020-08-05/001/alf/probe00
# 22 b749446c-18e3-4987-820a-50649ab0f826 aad23144-0e52-4eac-80c5-c4ee2decb198 /mnt/s0/Data/cortexlab/Subjects/KS023/2019-12-10/001/alf/probe01
# 23 36362f75-96d8-4ed4-a728-5e72284d0995 d0ea3148-948d-4817-94f8-dcaf2342bbbe /mnt/s0/Data/mainenlab/Subjects/ZFM-01936/2021-01-19/001/alf/probe00
# 24 9657af01-50bd-4120-8303-416ad9e24a51 7f6b86f9-879a-4ea2-8531-294a221af5d0 /mnt/s0/Data/zadorlab/Subjects/CSH_ZAD_019/2020-08-14/001/alf/probe00
# 25 dab512bd-a02d-4c1f-8dbc-9155a163efc0 d23a44ef-1402-4ed7-97f5-47e9a7a504d9 /mnt/s0/Data/danlab/Subjects/DY_016/2020-09-12/001/alf/probe00


pids = ["ce397420-3cd2-4a55-8fd1-5e28321981f4",
        "e31b4e39-e350-47a9-aca4-72496d99ff2a",
        "f8d0ecdc-b7bd-44cc-b887-3d544e24e561",
        "6fc4d73c-2071-43ec-a756-c6c6d8322c8b",
        "c17772a9-21b5-49df-ab31-3017addea12e",
        "0851db85-2889-4070-ac18-a40e8ebd96ba",  # this one has no pykilosort ...
        "eeb27b45-5b85-4e5c-b6ff-f639ca5687de",
        "69f42a9c-095d-4a25-bca8-61a9869871d3",
        "f03b61b4-6b13-479d-940f-d1608eb275cc",
        "f2ee886d-5b9c-4d06-a9be-ee7ae8381114",
        "f26a6ab1-7e37-4f8d-bb50-295c056e1062",
        "c4f6665f-8be5-476b-a6e8-d81eeae9279d",
        "9117969a-3f0d-478b-ad75-98263e3bfacf",
        "febb430e-2d50-4f83-87a0-b5ffbb9a4943",
        "8413c5c6-b42b-4ec6-b751-881a54413628",
        "8b7c808f-763b-44c8-b273-63c6afbc6aae",
        "f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c",
        "63517fd4-ece1-49eb-9259-371dc30b1dd6",
        "8d59da25-3a9c-44be-8b1a-e27cdd39ca34",
        "19baa84c-22a5-4589-9cbd-c23f111c054c",
        "143dd7cf-6a47-47a1-906d-927ad7fe9117",
        "84bb830f-b9ff-4e6b-9296-f458fb41d160",
        "b749446c-18e3-4987-820a-50649ab0f826",
        "36362f75-96d8-4ed4-a728-5e72284d0995",
        "9657af01-50bd-4120-8303-416ad9e24a51",
        "dab512bd-a02d-4c1f-8dbc-9155a163efc0"]

INDEX = 3  # let's say we're interested in that one
pid = pids[INDEX]
eid, pname = one.pid2eid(pid)
session_path = one.eid2path(eid)

pyks = {}
pyks['spikes'] = one.load_object(eid, 'spikes', collection=f'alf/{pname}/pykilosort')
pyks['clusters'] = one.load_object(eid, 'clusters', collection=f'alf/{pname}/pykilosort')
pyks['channels'] = one.load_object(eid, 'channels', collection=f'alf/{pname}/pykilosort')

ks2 = {}
ks2['spikes'] = one.load_object(eid, 'spikes', collection=f'alf/{pname}')
ks2['clusters'] = one.load_object(eid, 'clusters', collection=f'alf/{pname}')
ks2['channels'] = one.load_object(eid, 'channels', collection=f'alf/{pname}')

ks2p = {}
ks2p['spikes'] = one.load_object(eid, 'spikes', collection=f'alf/{pname}/ks2_preproc_tests')
ks2p['clusters'] = one.load_object(eid, 'clusters', collection=f'alf/{pname}/ks2_preproc_tests')
ks2p['channels'] = one.load_object(eid, 'channels', collection=f'alf/{pname}/ks2_preproc_tests')
