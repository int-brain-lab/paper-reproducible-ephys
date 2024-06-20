"""
Methods:

 N=xx adult mice (C57BL/6, male and female, obtained from either Jackson Laboratory or Charles River)
 were used in this study. Mice were aged XX–XX weeks and weighed XX–XX g on the day
 of electrophysiological recording.
"""
from one.api import ONE
from reproducible_ephys_functions import query
from datetime import datetime

one = ONE()
q = query()

# Take diff between RE recording date and subj DOB
# sub_list = list()

sub_dict = dict()
id_subj_all = list()
for q_i in q:
    # Session date
    date_sess = q_i['session']['start_time'][0:10]
    date_format = '%Y-%m-%d'  # Transform string into date value
    date_sess = datetime.strptime(date_sess, date_format)

    # Get subject for this session
    subj = one.alyx.rest('subjects', 'list',
                         nickname=q_i['session']['subject'],
                         lab=q_i['session']['lab'])[0]
    # DOB of subject
    date_subj = subj['birth_date']

    # Save date into subj for checking
    subj['date_sess_analysis'] = date_sess  # For checking
    subj['dob_analysis'] = date_subj  # For checking

    if date_subj is None:
        print(f"This subject has no DOB: {subj['nickname']}")
        continue
    date_format = '%Y-%m-%d'  # Transform string into date value
    date_subj = datetime.strptime(date_subj, date_format)

    # Take difference
    date_diff = date_sess - date_subj

    id_subj = subj['id']
    if id_subj not in id_subj_all:  # Remove duplicates if any
        subj['date_diff_re'] = date_diff
        # sub_list.append(subj)
        sub_dict[id_subj] = subj
        id_subj_all.append(id_subj)
    else:  # There are multiple sessions for the subj. Check the date and update if smaller
        if date_diff < sub_dict[id_subj]['date_diff_re']:
            # find subject in the dict and replace value
            subj['date_diff_re'] = date_diff
            sub_dict[id_subj] = subj

##
sub_list = sub_dict.values()

# AGE RANGE
age_weeks = [item['date_diff_re'] for item in sub_list]
print(f'AGE RANGE: {min(age_weeks)}-{max(age_weeks)} weeks')

# WEIGHT RANGE
ref_weight = [item['reference_weight'] for item in sub_list]
print(f'WEIGHT RANGE: {min(ref_weight)}-{max(ref_weight)} g')

##
# Print as age seems large
for subj in sub_list:
    print(f'{subj["nickname"]} - DOB: {subj["dob_analysis"]} analysis / {subj["birth_date"]} alyx str, '
          f'\n \t \t session: {subj["date_sess_analysis"]} , Diff: {subj["date_diff_re"]}')

##
# This section is commented out because 'age_weeks' is not a good value to use
# sub_list = list()
# id_subj_all = list()
# for q_i in q:
#     subj = one.alyx.rest('subjects', 'list',
#                          nickname=q_i['session']['subject'],
#                          lab=q_i['session']['lab'])[0]
#     id_subj = subj['id']
#     if id_subj not in id_subj_all:  # Remove duplicates if any
#         sub_list.append(subj)
#         id_subj_all.append(id_subj)
#
# # N SUBJECT
# print(f'N subject: {len(sub_list)}')
#
# # AGE RANGE
# age_weeks = [item['age_weeks'] for item in sub_list]
# print(f'AGE RANGE: {min(age_weeks)}-{max(age_weeks)} weeks')
#
# # WEIGHT RANGE
# ref_weight = [item['reference_weight'] for item in sub_list]
# print(f'WEIGHT RANGE: {min(ref_weight)}-{max(ref_weight)} g')
#
# ##
# # TODO the below should not be necessary,
# #  but there is 1 mouse where age range is 0
# import numpy as np
# age_weeks = np.array(age_weeks)
# indx = np.where(age_weeks > 0)[0]
# print(f'AGE RANGE: {min(age_weeks[indx])}-{max(age_weeks)} weeks')
#
# # Find mouse with age 0
# nicknames = np.array([item['nickname'] for item in sub_list])
# indx = np.where(np.array(age_weeks) == 0)[0]
# print(f'Mouse with 0 age: {nicknames[indx]}')

##

# This does not work as not all animals have key ready4recording
# date_ready4recording = [item['json']['trained_criteria']['ready4recording'][0] for item in sub_list]

# for subj in sub_list:
#
#     if 'trained_criteria' not in subj['json'].keys():
#         print(f'No trained_criteria {subj["nickname"]}')
#         continue
#
#     if 'ready4recording' not in subj['json']['trained_criteria'].keys():
#         print(f'No ready4recording {subj["nickname"]}')

