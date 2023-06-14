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


def normalize_atoms(atom):
    return re.sub("\d+", "", atom)

def cif_parser(cif_path, primitive=False):
    """
    Parser for single cif file
    """
    s = Structure.from_file(cif_path, primitive=primitive)
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
            'abc_coordinates':abc_coordinates
            }

def single_parser(content):
    dir_path = '/mof_database' # replace to your MOF database path
    cif_name, targets = content
    cif_path = os.path.join(dir_path, cif_name+'.cif')
    if os.path.exists(cif_path):
        data =  cif_parser(cif_path, primitive=False)
        data['target'] = targets
        return pickle.dumps(data, protocol=-1)
    else:
        print(f'{cif_path} does not exit!')
        return None

def get_data(path):
    data = pd.read_csv(path)
    columns = 'target' # replace to your target column
    cif_names = 'mof-name' # replace to your mof name column

    value = data[columns]
    _mean,_std = value.mean(), value.std()
    print(f'mean and std of target values are: {_mean}, {_std}')

    return [(item[0], item[1]) for item in zip(data[cif_names], data[columns].values)]

def train_valid_test_split(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    np.random.seed(42)
    id_list = [item[0] for item in data]
    unique_id_list = list(set(id_list))
    unique_id_list = np.random.permutation(unique_id_list)
    print(f'length of data is {len(data)}')
    print(f'length of unique_id_list is {len(unique_id_list)}')
    train_size = int(len(unique_id_list) * train_ratio)
    valid_size = int(len(unique_id_list) * valid_ratio)
    train_id_list = unique_id_list[:train_size]
    valid_id_list = unique_id_list[train_size:train_size+valid_size]
    test_id_list = unique_id_list[train_size+valid_size:]

    train_data = [item for item in data if item[0] in train_id_list]
    valid_data = [item for item in data if item[0] in valid_id_list]
    test_data = [item for item in data if item[0] in test_id_list]

    print(f'train_len:{len(train_data)}')
    print(f'valid_len:{len(valid_data)}')
    print(f'test_len:{len(test_data)}')

    return train_data, valid_data, test_data

def write_lmdb(inpath='./', outpath='./',nthreads=40):
    data = get_data(inpath)
    train_data, valid_data, test_data = train_valid_test_split(data)
    print(len(train_data), len(valid_data), len(test_data))
    for name, content in [ ('train.lmdb', train_data), 
                            ('valid.lmdb', valid_data), 
                            ('test.lmdb', test_data) ]:
        outputfilename = os.path.join(outpath, name)
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
