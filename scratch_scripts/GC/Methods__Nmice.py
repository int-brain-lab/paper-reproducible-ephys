"""
Methods:

 N=xx adult mice (C57BL/6, male and female, obtained from either Jackson Laboratory or Charles River)
 were used in this study. Mice were aged XX–XX weeks and weighed XX–XX g on the day
 of electrophysiological recording.
"""
from one.api import ONE
from reproducible_ephys_functions import query

one = ONE()
q = query()

sub_list = list()
id_subj_all = list()
for q_i in q:
    subj = one.alyx.rest('subjects', 'list',
                         nickname=q_i['session']['subject'],
                         lab=q_i['session']['lab'])[0]
    id_subj = subj['id']
    if id_subj not in id_subj_all: # Remove duplicates if any
        sub_list.append(subj)
        id_subj_all.append(id_subj)

# N SUBJECT
print(f'N subject: {len(sub_list)}')

# AGE RANGE
age_weeks = [item['age_weeks'] for item in sub_list]
print(f'AGE RANGE: {min(age_weeks)}-{max(age_weeks)} weeks')

# WEIGHT RANGE
ref_weight = [item['reference_weight'] for item in sub_list]
print(f'WEIGHT RANGE: {min(ref_weight)}-{max(ref_weight)} g')
