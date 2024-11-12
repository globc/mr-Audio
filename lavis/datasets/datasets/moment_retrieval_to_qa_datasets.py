import os

import torch

from lavis.datasets.datasets.base_dataset import BaseDataset

import os
from collections import OrderedDict



class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )

class MomentRetrievalForQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # set video clip if 'start'&'end' timestamp in data
        if "start" in ann:
            start, end = float(ann["start"]), float(ann["end"])
            # start, end = int(float(ann["start"]) * 100), int(float(ann["end"]) * 100)
            clip = [start, end]
        else:
            clip = None

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname + ".mp4")

        frms, indices, fps = self.vis_processor(video_path, clip_proposal=clip)
        query = ann["query"]
        relevant_windows = str(ann["relevant_windows"])

        query_prompt = "Query: " + query + "\n"
        task_prompt = "Given the video and the query, find the relevant windows.\nRelevant windows: "

        # generate video prompt in the following format:
        # <vid><t><t+1><t+2>…<duration>[frame embeddings]</vid>
        # where <vid> is the video id, and <t> are the timestamps of each frame
        frms = frms.permute(1, 0, 2, 3)
        time_stamps = [float(idx / fps) for idx in indices]
        duration = ann["duration"]

        timestamps = [round(t, 2) for t in time_stamps]
        # timestamps.append(duration)

        timestamps = torch.tensor(timestamps)

        duration = torch.tensor(duration)

        question = query_prompt + task_prompt

        answer = relevant_windows + ann["answer"]

        return {
            "video": frms,
            "text_input": question,
            "answer": answer,
            "text_output": answer,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "weight": [1.]
        }
