from utils.import_head import global_params
from utils.save_and_load_data import read_Xy_encoding2, load_df_from_Xy_regression_setup
from utils.train_and_load_model import load_RRRglobal_res
from utils.utils import load_neuron_nparray, make_folder

import os, pdb
import numpy as np

### load data
gp_setup = dict(wa = 'repsite', vt='clean', it=f"good_unit_Xstandardized")
gp = global_params(which_areas=gp_setup['wa'], var_types=gp_setup['vt'], inc_type=gp_setup['it'])
resgood_folder = make_folder("./result")

Xy_regression = read_Xy_encoding2(gp)
data_df = load_df_from_Xy_regression_setup(['mfr_task'], Xy_regression)
RRR_res_df = load_RRRglobal_res(gp)
RRR_res_df = RRR_res_df.merge(data_df, left_on=['eid', 'ni'], right_on=['eid', 'ni'])

inc_param=dict(min_r2=0.03, sel="U"); 

if inc_param['sel'] == "U":
    coef_vs = load_neuron_nparray(RRR_res_df, "RRRglobal_U")
    coef_vs = coef_vs.reshape((coef_vs.shape[0],-1))
    print(coef_vs.shape)

eids_list = np.sort(RRR_res_df['eid'].unique())
from one.api import ONE
cache_folder = "./data/cache/"
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', 
          cache_dir=cache_folder)
labs_list = [one.eid2pid(one.pid2eid(pid)[0], details=True)[2][0]['session_info']['lab'] for pid in eids_list]
print(labs_list)
# labs_list = ['mainenlab', 'churchlandlab_ucla', 'cortexlab', 'churchlandlab_ucla', 'zadorlab', 'churchlandlab_ucla', 'cortexlab', 'churchlandlab_ucla', 'mainenlab', 'mrsicflogellab', 'steinmetzlab', 'steinmetzlab', 'cortexlab', 'mrsicflogellab', 'danlab', 'churchlandlab', 'churchlandlab', 'angelakilab', 'danlab', 'mrsicflogellab', 'cortexlab', 'churchlandlab_ucla', 'angelakilab', 'steinmetzlab', 'churchlandlab_ucla', 'churchlandlab_ucla', 'steinmetzlab', 'hoferlab', 'angelakilab', 'mainenlab', 'wittenlab', 'mainenlab', 'angelakilab', 'cortexlab', 'angelakilab', 'mrsicflogellab', 'zadorlab', 'churchlandlab_ucla', 'hoferlab', 'mainenlab', 'cortexlab', 'churchlandlab_ucla', 'steinmetzlab', 'mainenlab', 'angelakilab', 'cortexlab', 'danlab', 'cortexlab', 'cortexlab', 'mainenlab', 'mainenlab', 'cortexlab', 'cortexlab', 'mainenlab', 'danlab', 'wittenlab', 'churchlandlab_ucla', 'wittenlab', 'hoferlab', 'danlab', 'churchlandlab_ucla', 'churchlandlab', 'churchlandlab_ucla', 'cortexlab', 'churchlandlab_ucla', 'churchlandlab', 'mainenlab', 'steinmetzlab', 'angelakilab', 'churchlandlab', 'wittenlab']
institution_map = {'cortexlab': 'UCL', 
                   'mainenlab': 'CCU', 
                   'zadorlab': 'CSHL (Z)', 
                   'churchlandlab': 'CSHL (C)', 
                   'angelakilab': 'NYU', 
                   'wittenlab': 'Princeton', 
                   'hoferlab': 'SWC', 
                   'mrsicflogellab': 'SWC', 
                   'danlab': 'Berkeley', 
                   'steinmetzlab': 'UW', 
                   'churchlandlab_ucla': 'UCLA', 
                   'hausserlab': 'UCL (H)'}
eid2labid = {eids_list[i]: institution_map[labs_list[i]] for i in range(len(eids_list))}
RRR_res_df['lab'] = RRR_res_df.eid.apply(lambda eid: eid2labid[eid])
RRR_res_df.loc[RRR_res_df.acronym=="VISa", 'acronym'] = "VISa_am"
RRR_res_df.loc[RRR_res_df.acronym=="VISam", 'acronym'] = "VISa_am"

np.savez(os.path.join(resgood_folder, "data.npz"), 
         coef_vs=coef_vs,
         r2s=RRR_res_df[f"RRRglobal_r2"].to_numpy(),
         eids=RRR_res_df["eid"].to_numpy(),
         labs=RRR_res_df["lab"].to_numpy(),
         acronyms=RRR_res_df["acronym"].to_numpy())




nis_incmask = (RRR_res_df["RRRglobal_r2"] >= 0.03)
print("number of ctx neruons:", len(nis_incmask))
print("number of selective ctx neruons:", np.sum(nis_incmask))
print("mean R2:", np.mean(RRR_res_df.loc[nis_incmask, "RRRglobal_r2"]))
print("# eids:", RRR_res_df["eid"].nunique())
print("# neurons:", RRR_res_df["acronym"].value_counts())
print("# eids:", RRR_res_df.groupby('acronym').eid.nunique())
print(RRR_res_df["eid"].unique())
