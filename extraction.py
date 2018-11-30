import numpy as np
from simpy import analyze
import os
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
            cr_x = vals[1]
            cr_ke = vals[4]

            hist_ke[prev_index:cr_index] = prev_ke
            hist_x[prev_index:cr_index] = cr_x - start_x

            prev_ke = cr_ke
            prev_index = cr_index

        hist_ke[cr_index:] = 0.0
        hist_x[cr_index:] = hist_x[cr_index-1]

        matrix_ke.append(hist_ke)
        matrix_x.append(hist_x)

    return np.array(matrix_ke), np.array(matrix_x), np.array(init_z)


def extract_sim(simdir, masterfile):
    simfiles = os.listdir(simdir)
    master_dict = extract_master(masterfile)

    ek_list = []
    dx_list = []
    z_list = []


    for simfile in simfiles:
        print("Processing: {}".format(simfile))
        simfile = "{}/{}".format(simdir, simfile)
        d, simlen = extract_dict(simfile, master_dict)
        ek, x, z = expand_dict(d, simlen)

        ek_list.append(np.max(ek, axis=1))
        dx_list.append(x[:,-1])
        z_list.append(z)

    final = [np.concatenate(ek_list),
             np.concatenate(dx_list),
             np.concatenate(z_list)]

    return final

#master_dict = extract_master('master_sample_hobler.lammpstrj')
#d, simlen = extract_dict('ek_100_theta_90_z_-43.8942676673338_id_30722.lammpstrj', master_dict)
#ek, x, z = expand_dict(d, simlen)

#mek = np.max(ek, axis=1)
#dx = x[:,-1]
final = extract_sim('100ek_60-50', 'master_sample_hobler_50-60.lammpstrj')
pickle.dump(final, open('100ek_90deg_-60A_-50A.pickle', 'wb'))
