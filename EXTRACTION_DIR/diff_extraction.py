import numpy as np
from simpy import analyze
import os
import re
import pickle

from collections import defaultdict
import matplotlib.pyplot as plt

analyze.get_logger().setLevel(40)

def extract_master(ifile):
    frame = analyze.Traj(ifile).read(1)[0]

    coords = frame['coord']
    ids = frame['id']

    mdict = {}
    for i, c in zip(ids, coords):
        mdict[i] = c

    return mdict

def extract_dict(ifile, mdict):
    filename = ifile.split('/')[-1]
    filename = ".".join(filename.split('.')[:-1])
    name_split = filename.split('_')
    if False:
        name_dict = dict(zip(name_split[0::2], name_split[1::2]))

        primal_id = int(name_dict['id'])
        primal_ke = float(name_dict['ek'])

    else:
        primal_id = int(name_split[3])
        primal_ke = 100

    frames = analyze.Traj(ifile).read(-1)
    atom_dict = defaultdict(list)
    # [ step, x, y, z,  ke, pe //, type ]
    # [   0 , 1, 2, 3,  4 , 5  //,  6   ]
    max_ke_dict = defaultdict(float)
    for step, f in enumerate(frames):
        for _id, coord, ke, pe, _type in zip(f['id'], f['coord'], f['c_kea'], f['c_pea'], f['type']):

            # check if id defined in current file
            if not atom_dict[_id]:
                if _id == primal_id:
                    _ke = primal_ke
                else:
                    _ke = 0.0
                _coord = mdict[_id]
                atom_line = [0, _coord[0], _coord[1], _coord[2], _ke, pe, _type]
                atom_dict[_id].append(atom_line)

            atom_line = [step, coord[0], coord[1], coord[2], ke, pe]
            atom_dict[_id].append(atom_line)

            max_ke_dict[_id] = max(ke, max_ke_dict[_id])

    # discard low energy 0.1

    for _id in max_ke_dict.keys():
        max_ke = max_ke_dict[_id]
        if max_ke <= 0.1:
            del atom_dict[_id]

    return atom_dict, step+1

def expand_dict(atom_dict, simlen, ptype=2):
    ids = sorted(atom_dict.keys())

    matrix_ke = []
    matrix_x = []
    init_z = []

    for _id in ids:
        vals = atom_dict[_id][0]

        if vals[6] != ptype:
            continue

        hist_ke = np.empty(simlen, dtype=float)
        hist_x = np.empty(simlen, dtype=float)

        init_z.append(vals[3])

        prev_ke = vals[4]
        start_x = vals[1]

        prev_index = 0
        for vals in atom_dict[_id][1:]:
            cr_index = vals[0]
            cr_ke = vals[4]
            cr_x = vals[1]

            hist_ke[prev_index:cr_index] = prev_ke
            hist_x[prev_index:cr_index] = cr_x - start_x

            prev_ke = cr_ke
            prev_index = cr_index

        hist_ke[cr_index:] = 0.0
        hist_x[cr_index:] = hist_x[cr_index-1]

        matrix_ke.append(hist_ke)
        matrix_x.append(hist_x)

    return np.array(matrix_ke), np.array(matrix_x), np.array(init_z)


def extract_sim(simfiles, masterfile):
    #simfiles = os.listdir(simdir)
    master_dict = extract_master(masterfile)

    ek_list = []
    dx_list = []
    z_list = []
    index_list = []

    sim_cnt = 0
    for simfile in simfiles:
        print("Processing: {}".format(simfile))
        d, simlen = extract_dict(simfile, master_dict)
        ek, x, z = expand_dict(d, simlen)

        _ek = np.max(ek, axis=1)
        _dx = x[:,-1]
        _z = z

        ek_list.append(_ek)
        dx_list.append(_dx)
        z_list.append(_z)
        index_list.append(sim_cnt*np.ones(len(_z)))

        sim_cnt += 1

    final = [np.concatenate(ek_list),
             np.concatenate(dx_list),
             np.concatenate(z_list),
             np.concatenate(index_list),
             sim_cnt]

    return final

#master_dict = extract_master('master_sample_hobler.lammpstrj')
#d, simlen = extract_dict('ek_100_theta_90_z_-43.8942676673338_id_30722.lammpstrj', master_dict)
#ek, x, z = expand_dict(d, simlen)


target_dir = "4500"

target_depth_dir = [depth for depth in os.listdir(target_dir) if re.match("^_", depth)]

master_sample = [m for m in os.listdir(target_dir) if re.match(".*lammpstrj", m)][0]
master_sample_path = "{}/{}".format(target_dir, master_sample)

target_depth_paths = [("{}/{}".format(target_dir, t), t) for t in target_depth_dir]

print(target_depth_paths)
print(master_sample_path)
#targets = ["{}/{}".format(base_dir, ldir) for ldir in os.listdir(base_dir)]
#print(targets)

for depth_path, depth_str in target_depth_paths:
    ek_str = "100ek"

    all_dumpfiles = os.listdir(depth_path)
    dumpfile_angle_dict = {"UP":[], "NORMAL":[], "DOWN":[]}
    theta_name_dict = {"30": "UP", "90": "NORMAL", "150": "DOWN"}

    for dfile in all_dumpfiles:
        theta_ = dfile.split("_")[3]
        direction_ = theta_name_dict[theta_]

        dumpfile_angle_dict[direction_].append("{}/{}".format(depth_path, dfile))


    for angle in dumpfile_angle_dict:
        print(depth_path, angle)
        sim_targets = dumpfile_angle_dict[angle]

        extracted_results = extract_sim(sim_targets, master_sample_path)

        pickle_file = "{}_{}_{}_{}.pickle".format(ek_str, depth_str, angle, target_dir)
        pickle.dump(extracted_results, open(pickle_file, 'wb'))

"""
name_prefix = "100ek_small_NOrelax"
for target in targets:
    final = extract_sim(target, 'master_small.lammpstrj')

    slice_name = target.split("/")[-1]
    name = "{}_{}.pickle".format(name_prefix, slice_name)
    out_file = "{}/{}".format("data_small_NOrelax", name)
    pickle.dump(final, open(out_file, 'wb'))
"""
