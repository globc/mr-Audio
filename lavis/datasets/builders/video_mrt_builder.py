"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset, VideoQAInstructDataset
from lavis.datasets.datasets.music_avqa import MusicAVQAInstructDataset, MusicAVQADataset
from lavis.datasets.datasets.mc_video_vqa_datasets import MCVideoQADataset


class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

    def build(self):
        datasets = super().build()

        ans2label = self.config.build_info.annotations.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)

        for split in datasets:
            datasets[split]._build_class_labels(ans2label)

        return datasets


class MCVideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = MCVideoQADataset
    eval_dataset_cls = MCVideoQADataset

    def build(self):
        datasets = super().build()

        for split in datasets:
            datasets[split]._load_auxiliary_mappings()

        return datasets


@registry.register_builder("msrvtt_mrt_instruct")
class MSRVTTQAInstructBuilder(VideoQABuilder):
    train_dataset_cls = VideoQAInstructDataset
    eval_dataset_cls = VideoQAInstructDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa_instruct.yaml",
    }


