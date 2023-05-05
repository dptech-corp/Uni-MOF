Official Repository for the Uni-MOF Series Methods
==================================================

Shortcuts
---------

- [Uni-MOF](./unimof/)


**Note**: if you want to install or run our codes, please `cd` to subfolders first.


Uni-MOF: A Universal 3D Material Representation Learning Framework for Gas Adsorption in MOFs
-------------------------------------------------------------------

[[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/6447d756e4bbbe4bbf3afeaa)]

Authors: Jingqi Wang, Jiapeng Liu, Hongshuai Wang, Guolin Ke, Linfeng Zhang, Jianzhong Wu, Zhifeng Gao, Diannan Lu

<p align="center"><img src="unimof/figure/overview_new.jpg" width=60%></p>
<p align="center"><b>Schematic illustration of the Uni-MOF framework</b></p>

Uni-MOF is an innovative framework for large-scale, three-dimensional MOF representation learning, designed for universal multi-gas prediction.  Specifically, Uni-MOF serves as a versatile "gas adsorption detector" for MOF materials, employing pure three-dimensional representations learned from over 631,000 collected MOF and COF structures.  Our experimental results show that Uni-MOF can automatically extract structural representations and predict adsorption capacities under various operating conditions using a single model.  For simulated data, Uni-MOF exhibits remarkably high predictive accuracy across all datasets.  Impressively, the values predicted by Uni-MOF correspond with the outcomes of adsorption experiments.  Furthermore, Uni-MOF demonstrates considerable potential for broad applicability in predicting a wide array of other properties.

Check this [subfolder](./unimof/) for more detalis.

Uni-MOF's data 
------------------------------

For the details of datasets, please refer to Table 1 in our [paper](https://chemrxiv.org/engage/chemrxiv/article-details/6447d756e4bbbe4bbf3afeaa).

There are total 6 datasets:


| Data                     | File Size  | Update Date | Download Link                                                                                                             | 
|--------------------------|------------| ----------- |---------------------------------------------------------------------------------------------------------------------------|
| nanoporous material pretrain | GB   | May 10 2023 |                                |
| gas adsorption property      | GB   | May 10 2023 |          |
| material structural property | GB   | May 10 2023 |                |

We use [LMDB](https://lmdb.readthedocs.io) to store data, you can use the following code snippets to read from the LMDB file.

```python
import lmdb
import numpy as np
import os
import pickle

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
```
We use pickle protocol 5, so Python >= 3.8 is recommended.

Uni-Mol's pretrained model weights
----------------------------------

| Model                     | File Size  |Update Date | Download Link                                                | 
|--------------------------|------------| ------------|--------------------------------------------------------------|
| nanoporous material pretrain   | MB   | May 10 2023 |https://github.com/dptech-corp/Uni-MOF/releases/download/     |


Uni-Mol's finetuned model weights
----------------------------------

| Model                                           | File Size| Update Date| Download Link                                                     | 
|-------------------------------------------------|---------| -----------|--------------------------------------------------------------------|
| hMOF_MOFX_DB         | MB   | May 10 2023 |https://github.com/dptech-corp/Uni-MOF/releases/download    |
| CoRE_MOFX_DB       | MB   | May 10 2023 |https://github.com/dptech-corp/Uni-Mol/releases/download  |
| CoRE_MAP_DB          | MB   | May 10 2023 |https://github.com/dptech-corp/Uni-Mol/releases/download      |


Citation
--------

Please kindly cite our papers if you use the data/code/model.
```
@article{wang2023metal,
  title={Metal-organic frameworks meet Uni-MOF: a revolutionary gas adsorption detector},
  author={Wang, Jingqi and Liu, Jiapeng and Wang, Hongshuai and Ke, Guolin and Zhang, Linfeng and Wu, Jianzhong and Gao, Zhifeng and Lu, Diannan},
  year={2023}
}
```

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/dptech-corp/Uni-MOF/blob/main/LICENSE) for additional details.
