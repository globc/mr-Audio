"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.moment_retrieval_to_qa_datasets import  MomentRetrievalForQADataset



class MomentRetrievalForQABuilder(BaseDatasetBuilder):
    train_dataset_cls = MomentRetrievalForQADataset
    eval_dataset_cls = MomentRetrievalForQADataset


    def build(self):
        datasets = super().build()

        ans2label = self.config.build_info.annotations.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)

        for split in datasets:
            datasets[split]._build_class_labels(ans2label)

        return datasets


@registry.register_builder("charades_mrt")
class CharadesToQAInstructBuilder(MomentRetrievalForQABuilder):
    train_dataset_cls = MomentRetrievalForQADataset
    eval_dataset_cls = MomentRetrievalForQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charade/defaults_cap_instruct.yaml",
    }


