# main.py

from flask import Blueprint, render_template
from flask_login import login_required, current_user

main = Blueprint('main', __name__)


cond1 = ['noncluster', 'og_nearest_eye.json', 'cluster', 'og_cluster_hat.json']
cond2 = ['noncluster', 'dr_nearest_eye.json', 'cluster', 'dr_cluster_hat.json']
cond3 = ['cluster', 'og_cluster_hat.json', 'noncluster', 'og_nearest_eye.json']
cond4 = ['cluster', 'dr_cluster_hat.json', 'noncluster', 'dr_nearest_eye.json']
cond5 = ['noncluster', 'og_nearest_hat.json', 'cluster', 'og_cluster_eye.json']
cond6 = ['noncluster', 'dr_nearest_hat.json', 'cluster', 'dr_cluster_eye.json']
cond7 = ['cluster', 'og_cluster_eye.json', 'noncluster', 'og_nearest_hat.json']
cond8 = ['cluster', 'dr_cluster_eye.json', 'noncluster', 'dr_nearest_hat.json']

conds = [cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8]
condition_ids = ['a','b','c','d','e','f','g','h']


def get_cid(x): 
    if x in condition_ids:
        return condition_ids.index(x)
    else:
        return 1
def cond_dict(cond_list):
    d = {
        "task1_page": cond_list[0],
        "task1_data": "static/" +cond_list[1],
        "task2_page": cond_list[2],
        "task2_data": "static/" +cond_list[3],
    }
    return d

def cond_dict_from_cid(cid):
    return cond_dict(conds[cid])




# home GET, login required
@main.route('/')
@main.route('/home')
@login_required
def home():
    #print(current_user.name[0])
    #print(dir(current_user))
    #print(dir(current_user.email))
    cid = get_cid(current_user.name[0])
    first_task = cond_dict_from_cid(cid)["task1_page"] 
    return render_template('home.html', email=current_user.email, name=current_user.name, id=current_user.id, first_task = first_task)


# study main page GET, login required
@main.route('/study1')
@login_required
def study1():
    data = cond_dict_from_cid(get_cid(current_user.name[0]))
    return render_template(data["task1_page"] + '.html', data_file= data["task1_data"], name=current_user.name)


@main.route('/study2')
@login_required
def study2():
    data = cond_dict_from_cid(get_cid(current_user.name[0]))
    return render_template(data["task2_page"] + '.html', data_file= data["task2_data"], name=current_user.name)
