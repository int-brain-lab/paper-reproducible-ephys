from utils.import_head import global_params
from utils.save_and_load_data import read_Xy_encoding2
from utils.train_and_load_model import get_RRRglobal_res


gp_setup = dict(wa = 'repsite', vt='clean', it=f"good_unit_Xstandardized")
gp = global_params(which_areas=gp_setup['wa'], var_types=gp_setup['vt'], inc_type=gp_setup['it'])


Xy_regression = read_Xy_encoding2(gp)

get_RRRglobal_res(Xy_regression, gp)

