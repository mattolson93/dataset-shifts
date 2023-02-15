
import json


group_ids = ['a','b','c','d','e','f','g','h']



cond1 = ['noncluster', 'og_nearest_eye.json', 'cluster', 'og_cluster_hat.json']
cond3 = ['cluster', 'og_cluster_hat.json', 'noncluster', 'og_nearest_eye.json']
cond5 = ['noncluster', 'og_nearest_hat.json', 'cluster', 'og_cluster_eye.json']
cond7 = ['cluster', 'og_cluster_eye.json', 'noncluster', 'og_nearest_hat.json']
cond2 = ['noncluster', 'dr_nearest_eye.json', 'cluster', 'dr_cluster_hat.json']
cond4 = ['cluster', 'dr_cluster_hat.json', 'noncluster', 'dr_nearest_eye.json']
cond6 = ['noncluster', 'dr_nearest_hat.json', 'cluster', 'dr_cluster_eye.json']
cond8 = ['cluster', 'dr_cluster_eye.json', 'noncluster', 'dr_nearest_hat.json']

conds = [cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8]


out = {}
for g, c in zip(group_ids,conds):
    d = {
        "task1_page": c[0] + ".html",
        "task1_data": c[1],
        "task1_form": "eye" if "eye" in c[1] else "hat",
        "task2_page": c[2] + ".html",
        "task2_data": c[3],
        "task2_form": "eye" if "eye" in c[3] else "hat",
    }
    out[g] = d
'''for space in ['og','dr']:
    for do_cluster in ['cluster','nearest']:
        for task in ['eye','hat']:
            d = {
                "task1_page": cluster_data,
                "task1_data": train_data
                "task2_page": cluster_data,
                "task2_data": train_data
            }'''
    

with open("conditions.json",'w') as f:
    json.dump(out, f, indent=1)

