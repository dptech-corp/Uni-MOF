## placeholder for preprocess.py
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
from multiprocessing import Process, Queue, Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import timeout_decorator
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

def car_parser(car_path):
    return None

@timeout_decorator.timeout(20)
def cif_parser(cif_path, primitive=True):
    """
    Parser for single cif file
    """
    id = cif_path.split('/')[-1][:-4]
    s = Structure.from_file(cif_path, primitive=primitive)
    # analyzer = SpacegroupAnalyzer(s)
    # sym_cell = analyzer.get_symmetrized_structure()
    # spacegroup_info = analyzer.get_space_group_info()   # e.g. ('Fm-3m', 225)
    # wyckoff_symbol = sym_cell.wyckoff_symbol()

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

    return pickle.dumps({'ID':id, 
            'atoms':atoms, 
            'coordinates':coordinates, 
            'abc':abc, 
            'angles':angles, 
            'volume':volume, 
            'lattice_matrix':lattice_matrix, 
            'abc_coordinates':abc_coordinates
            }, protocol=-1)

def single_parser(cif_path):
    try:
        return cif_parser(cif_path, primitive=True)
    except:
        return None

def collect_cifs():
    cif_paths = []
    dir_path = './20220718MOF_database'
    choose_names = ['primitive_cif','CoRE', 'GA_MOFs', 'tobacco', 'tobacco_gen_30w', 'COF', 'qmof_database']
    for name in choose_names:
        if name == 'CoRE':
            paths = os.path.join(dir_path, name, 'structure_10143', '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        elif name == 'GA_MOFs':
            paths = os.path.join(dir_path, name, '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        elif name == 'primitive_cif':
            paths = os.path.join(dir_path, name, '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        elif name == 'tobacco':
            paths = os.path.join(dir_path, name, 'tobacco_1.0', 'cifs', '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        elif name == 'tobacco_generated_23w':
            paths = os.path.join(dir_path, name, '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        elif name == 'tobacco_generated_clean_15w':
            paths = os.path.join(dir_path, name, '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        elif name == 'COF':
            for sub_name in ['Bio_COFs_cif','CoRE_COF_cif','Functional_Design_cif']:
                paths = os.path.join(dir_path, name, sub_name, '*.cif')
                print(name, sub_name, len(glob.glob(paths)))
                cif_paths.extend(glob.glob(paths))
        elif name == 'tobacco_gen_30w':
            paths = os.path.join(dir_path, name, '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        elif name == 'qmof_database':
            paths = os.path.join(dir_path, name, 'relaxed_structures', '*.cif')
            print(name, len(glob.glob(paths)))
            cif_paths.extend(glob.glob(paths))
        else:
            raise ValueError('Unknown name: {}'.format(name))
    return cif_paths 

def write_lmdb(outpath='./', nthreads=40):
    cif_paths = collect_cifs()
    np.random.seed(42)
    cif_paths = np.random.permutation(cif_paths)
    train_ratio, val_ratio = 0.98, 0.02
    val_cifs = cif_paths[:int(len(cif_paths)*val_ratio)]
    train_cifs = cif_paths[int(len(cif_paths)*val_ratio):]
    for name, cifs in [('valid.lmdb', val_cifs),  \
                        ('train.lmdb',train_cifs)]:
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
            map_size=int(10e12),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(single_parser, cifs), total=len(cifs)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
                    if i % 1000 == 0:
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()

def token_collect():
    dir_path = './'
    outputfilename = os.path.join(dir_path, 'train.lmdb')
    env = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(10e9),
        )
    txn = env.begin()
    atoms_collects = []
    _keys = list(txn.cursor().iternext(values=False))
    for idx in tqdm(range(len(_keys))):
        datapoint_pickled = txn.get(f'{idx}'.encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        atoms_collects.extend(data['atoms'])
    atoms_collects = pd.Series(atoms_collects).value_counts()
    atoms_collects.to_csv('atoms_counts.csv',index=True, header=False)

if __name__ == '__main__':
    write_lmdb(outpath='./0302_lmdb', nthreads = int(sys.argv[1]))
    # print(single_parser('/mnt/vepfs/gaozhifeng/unimat_dev/examples/mof/20220718MOF_database/CoRE/structure_10143/ADASAB_charged.cif'))
    token_collect()
