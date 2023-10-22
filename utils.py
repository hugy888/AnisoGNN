import pickle
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

def mean_maxARE(y_pred, y_true):
    '''Calcuate Mean, MaxARE
    '''
    meanARE = np.around(100*np.mean((np.abs(y_pred-y_true)/y_true)), decimals=2)
    maxARE = np.around(100*np.max((np.abs(y_pred-y_true)/y_true)), decimals=2)
    return meanARE, maxARE


def seed_worker(worker_id):
    '''Seeding for DataLoaders
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(42)
    random.seed(42)


def get_data(eval, b_size, case):
    '''Prepare data depending on Eval Case

    Parameters
    ----------
    eval : {int}
        Range 1-4, indicating type of evalution/experiment

    b_size : {int}
        Batch size

    Returns
    -------
    train_loader : {DataLoader}
    test_loader : {DataLoader} - For test or validation data
    '''
    g = torch.Generator()
    g.manual_seed(42)

    train_data = []
    test_data = []
    all_data=[]

    if eval == 'test_45_90_deg':
        for texture in ['comp','uni','shear','psc']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
        for texture in ['comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    elif eval == 'test_45_deg':
        for texture in ['comp','uni','shear','psc','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
        for texture in ['comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    elif eval == 'test_ran_30%':
        np.random.seed(42)
        random.seed(42)
        rand_ids = np.random.choice(300, 300, replace=False)
        for texture in ['comp','uni','shear','psc','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90','comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']:
                with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                    data = pickle.load(handle)
                    all_data += data
        
        datalist_new = [all_data[i] for i in rand_ids] # shuffle
        # 70/30 split
        train_data=datalist_new[90:] 
        test_data=datalist_new[:90]

    elif eval == 'test_90_deg':
        for texture in ['comp','uni','shear','psc','comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
        for texture in ['comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    # Data Loaders
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True, worker_init_fn=seed_worker,
                              generator=g)
    test_loader = DataLoader(
        test_data, batch_size=len(test_data), shuffle=False, worker_init_fn=seed_worker,
        generator=g)

    return train_loader, test_loader

def scalers(case):
    # Re-scaling Predictions
    with open(f'./graph_data/{case}/y_scaler.pickle', 'rb') as handle:
        y_scaler = pickle.load(handle)
    with open(f'./graph_data/{case}/x_scaler.pickle', 'rb') as handle:
        x_scaler = pickle.load(handle)
    scaler = {}
    scaler['y'] = y_scaler
    scaler['x'] = x_scaler
    return scaler