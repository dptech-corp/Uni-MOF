from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
from multiprocessing import Process, Queue, Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np 
import pickle
import lmdb
import sys
import glob
import os
import re
from sklearn.preprocessing import MinMaxScaler
import itertools
import pdb
import random

def normalize_atoms(atom):
    return re.sub("\d+", "", atom)

def transform(cif_path):
    max_tolerance = 100
    s = CifParser(cif_path, occupancy_tolerance=max_tolerance)
    trans = ConventionalCellTransformation()
    s_trans = trans.apply_transformation(s.get_structures()[0])
    return s_trans

def cif_parser(cif_path, primitive=False):
    """
    Parser for single cif file
    """
    try:
        s = Structure.from_file(cif_path, primitive=primitive)
    except:
        s = transform(cif_path)
    id = cif_path.split('/')[-1][:-4]
    lattice = s.lattice
    abc = lattice.abc # lattice vectors
    angles = lattice.angles # lattice angles
    volume = lattice.volume # lattice volume
    lattice_matrix = lattice.matrix # lattice 3x3 matrix

    df = s.as_dataframe()
    atoms = df['Species'].astype(str).map(normalize_atoms).tolist()
    coordinates = df[['x', 'y', 'z']].values.astype(np.float32)
    abc_coordinates = df[['a', 'b', 'c']].values.astype(np.float32)
    assert len(atoms) == coordinates.shape[0]
    assert len(atoms) == abc_coordinates.shape[0]

    return {'ID':id, 
            'atoms':atoms, 
            'coordinates':coordinates, 
            'abc':abc, 
            'angles':angles, 
            'volume':volume, 
            'lattice_matrix':lattice_matrix, 
            'abc_coordinates':abc_coordinates,
            }

def single_parser(content):
    dir_path = '/mof_database' # replace to your MOF database path
    cif_name, gas, gas_attr, temperature, pressure, targets, task_name = content
    cif_path = os.path.join(dir_path, cif_name+'.cif')
    if os.path.exists(cif_path):
        data = cif_parser(cif_path, primitive=False)
        data['gas'] = np.array(gas, dtype=np.int32)
        data['gas_attr'] = gas_attr.astype(np.float32)
        data['temperature'] = np.array(temperature, dtype=np.float32)
        data['pressure'] = np.array(np.log10(pressure), dtype=np.float32)
        data['target'] = np.array(targets, dtype=np.float32)
        data['task_name'] = task_name
        return pickle.dumps(data, protocol=-1)
    else:
        print(f'{cif_path} does not exit!')
        return None

def get_data(path):
    data = pd.read_csv(path)
    cif_names = 'name'
    gas = 'gas-name'
    gas_attr = ['gas-CriticalTemp', 'gas-CriticalPressure', 'gas-AcentricFactor', 'gas-MolecularWeight', 'gas-MeltPoint', 'gas-BoilingPoint']
    temperature = 'temperature'
    pressure = 'pressure'
    columns = 'loading_[cm^3(STP)/gr(framework)]_abs_num'
    data['task_name'] = data[cif_names].astype(str) + '#' + data[gas].astype(str) + '#' + data[temperature].astype(str) + '#' + data[pressure].astype(str)

    # print mean and std
    value_log1p = np.log1p(data[columns])
    _mean,_std = value_log1p.mean(), value_log1p.std()
    print(f'mean and std of target values are: {_mean}, {_std}')

    return [(item[0], item[1], item[2], item[3], item[4], item[5], item[6]) for item in zip(data[cif_names], data[gas], data[gas_attr].values, data[temperature], data[pressure], data[columns], data['task_name'])]

# split the database into train, validation and test set according to gases
def train_valid_test_split(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    np.random.seed(42)
    id_list = [item[1] for item in data] ##gas_id
    unique_id_list = list(set(id_list))
    unique_id_list = np.random.permutation(unique_id_list)
    print(f'length of data is {len(data)}')
    print(f'length of unique_id_list is {len(unique_id_list)}')

    gas_dic = {1:"CH4", 2:"CO2", 3:"Ar", 4:"Kr", 5:"Xe", 6:"O2", 7:"N2"}
    gas_list = [1,2,3,4,5,6,7]

    print("*******************************")

    test_mat_id = 6
    test_id_list = np.array([test_mat_id])
    print("test_id_list:", gas_dic[test_mat_id])
    
    gas_list.remove(test_mat_id)
    valid_mat_id = random.sample(gas_list,1)
    valid_id_list = np.array([valid_mat_id])
    print("valid_id_list:", gas_dic[valid_mat_id[0]])

    gas_list.remove(valid_mat_id[0])
    train_id_list = np.array(gas_list)
    print("train_id_list:", train_id_list)

    train_data = [item for item in data if item[1] in train_id_list]
    valid_data = [item for item in data if item[1] in valid_id_list]
    test_data = [item for item in data if item[1] in test_id_list]

    print("*******************************")
    print(f'train_len:{len(train_data)}')
    print(f'valid_len:{len(valid_data)}')
    print(f'test_len:{len(test_data)}')

    return train_data, valid_data, test_data

def write_lmdb(inpath='./', outpath='./', nthreads=40):
    data = get_data(inpath)
    train_data, valid_data, test_data = train_valid_test_split(data)
    print(len(train_data), len(valid_data), len(test_data))
    for name, content in [ ('train.lmdb', train_data), 
                            ('valid.lmdb', valid_data), 
                            ('test.lmdb', test_data) ]:
        outputfilename = os.path.join(outpath, name)
        os.makedirs(os.path.dirname(outputfilename), exist_ok=True)
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(single_parser, content), total=len(content)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
                    if i % 1000 == 0:
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()

if __name__ == '__main__':

    inpath = '/data.csv' # replace to your data path
    outpath = '/outpath' # replace to your out path
    write_lmdb(inpath=inpath, outpath=outpath, nthreads=8)
