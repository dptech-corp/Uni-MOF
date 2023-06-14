# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    RawArrayDataset,
    RawNumpyDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
)
from unimat.data import (
    KeyDataset,
    LMDBDataset,
    ToTorchDataset,
    MaskPointsDataset,
    DistanceDataset,
    EdgeTypeDataset,
    PrependAndAppend2DDataset,
    RightPadDatasetCoord,
    LatticeNormalizeDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
    NumericalTransformDataset,
)
from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils

logger = logging.getLogger(__name__)

@register_task("unimof_v2")
class UniMOFV2Task(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path"
        )
        parser.add_argument(
            "--task-name",
            type=str,
            default='',
            help="downstream task name"
        )
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name"
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers"
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--finetune-mol-model",
            help="load unimat finetune model",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        tgt_dataset = KeyDataset(dataset, "target")
        tgt_dataset = ToTorchDataset(tgt_dataset, dtype='float32')
        if self.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates")
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.args.max_atoms)
        dataset = NormalizeDataset(dataset, "coordinates")
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        coord_dataset = KeyDataset(dataset, "coordinates")

        gas = KeyDataset(dataset, "gas")
        gas_attr = KeyDataset(dataset, "gas_attr")
        pressure = KeyDataset(dataset, "pressure")
        temperature = KeyDataset(dataset, "temperature")
        task_name = KeyDataset(dataset, "task_name")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = ToTorchDataset(coord_dataset, 'float32')
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.dictionary.pad(),
                        ),
                        "src_coord": RightPadDatasetCoord(
                            coord_dataset,
                            pad_idx=0,
                        ),
                        "src_distance": RightPadDataset2D(
                            distance_dataset,
                            pad_idx=0,
                        ),
                        "src_edge_type": RightPadDataset2D(
                            edge_type,
                            pad_idx=0,
                        ),
                        "gas": RawNumpyDataset(gas),
                        "gas_attr": RawNumpyDataset(gas_attr),
                        "temperature": RawNumpyDataset(temperature),
                        "pressure": RawNumpyDataset(pressure),
                    },
                    "task_name": RawArrayDataset(task_name),
                    "target": {
                        "finetune_target": tgt_dataset,
                    },
                },
            )
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.finetune_mol_model is not None:
                print("load pretrain model weight from...", args.finetune_mol_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_mol_model,
                )
                model.unimat.load_state_dict(state["model"], strict=False)
        return model