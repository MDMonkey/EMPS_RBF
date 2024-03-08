import pandas as pd
import numpy as np
import scipy as sp

from sklearn.model_selection import train_test_split
from icecream import ic

import os
import pathlib

def read_mat_emps(args):
    mat = sp.io.loadmat(args.PATH_DATA)
    Force1 = mat['Force1']
    size = int(len(Force1))
    print(size)
    input = Force1[:size].reshape(-1)
    y_data = mat['q_f'][:size]
    dy_data = mat['dq_f'][:size]
    output = np.concatenate((y_data, dy_data), axis=1)
    t = mat['td'][:size]
    return t, input, output

def read_mat_emps2(args):
    mat = sp.io.loadmat(args.PATH_DATA)
    ic(mat)
    Force1 = mat['Force1']
    size = int(len(Force1))
    print(size)
    input = Force1[:size].reshape(-1)
    y_data = mat['q_f'][:size]
    dy_data = mat['dq_f'][:size]
    output = np.concatenate((y_data, dy_data), axis=1)
    t = mat['td'][:size]
    return t, input, output

def read_mat_emps_test(path_mat_file):
    file_mat = sp.io.loadmat(path_mat_file)
    Force1 = file_mat['vir']*file_mat['gtau']
    t = file_mat['t']
    q_data = file_mat['qm']
    return t, Force1, q_data


def import_data(args, train = True):
    if train:
        file_path = args.PATH_TRAIN
    else:
        file_path = args.PATH_TEST
    
    t, inputs, outputs = read_mat_emps_test(file_path)

    if args.NORMALIZE_INPUT:
        inputs[:] = (inputs[:] - np.mean(inputs[:], axis=0)) / np.std(inputs[:],axis=0)
    
    return t, inputs, outputs



def import_data_(args):
    # Import the data
    t, inputs, outputs = read_mat_emps_test(args)

    if args.NORMALIZE_INPUT:
        inputs[:] = (inputs[:] - np.mean(inputs[:], axis=0)) / np.std(inputs[:],axis=0)
    
    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=args.VAL_SPLIT)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=args.TEST_SPLIT/(args.TEST_SPLIT + args.VAL_SPLIT)) 

    return t, x_train, x_val, x_test, y_train, y_val, y_test 

def make_files_location():
    ini_loc = pathlib.Path('../').parent.resolve()
    dir_res = os.path.join(ini_loc, "Results2_")
    if not os.path.isdir(dir_res):
        os.makedirs(dir_res)
        loc_res = os.path.join(dir_res, 'Test_11')
    else:
        existing = [int(name[-2:]) for name in os.listdir(dir_res)]
        loc_res  =os.path.join(dir_res, f'Test_{max(existing)+1}')
    os.makedirs(loc_res)
    return loc_res
    