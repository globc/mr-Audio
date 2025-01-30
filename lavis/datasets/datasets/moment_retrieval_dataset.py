import os

import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.processors.blip_processors import Blip2VideoTrainProcessor, BlipVideoEvalProcessor


class MomentRetrievalDataset(BaseDataset):
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

        video_prompt_end = "<extra_id_0>"

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": frms,
            "duration": duration,
            "query_id": ann["qid"],
            "timestamps": timestamps,
            "video_prompt_end": video_prompt_end,
            "query_prompt": query_prompt,
            "task_prompt": task_prompt,
            "relevant_windows": relevant_windows,
        }

class MomentRetrievalDataset_Audio(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['video_processor'], kwargs['text_processor'], kwargs['video_root'], kwargs['ann_paths'])
        self.audio_processor = kwargs['audio_processor'] if "audio_processor" in kwargs else None

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname + ".mp4")

        if "start" in ann:
            start, end = float(ann["start"]), float(ann["end"])

            try:
                import ffmpeg
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.filter(stream, 'crop', start=start, end=end)
                output_path = os.path.join(self.vis_root, f"{ann['video']}_clipped.mp4")
                ffmpeg.output(stream, output_path)
                ffmpeg.run(stream,overwrite_output=True)
                video_path = output_path
            except:
                print("video read error")
                video_path = None

        video, indices, fps = self.vis_processor(video_path)
        if isinstance(self.vis_processor, (Blip2VideoTrainProcessor, BlipVideoEvalProcessor)):
            video = video.permute(1, 0, 2, 3)
        timestamps = [round(idx / fps) for idx in indices]
        timestamps = torch.tensor(timestamps)

        duration = torch.tensor(ann["duration"])

        query = ann["query"]  

        example = """"
                    query: <Query> some military patriots takes us through their safety procedures and measures. <Query> 
                    duration: <Duration> 150 </Duration>
                    relevant_windows: [[0.80, 0.83], [0.84, 0.94]]', 

                    query: <Query> Man in baseball cap eats before doing his interview. <Query> 
                    duration:  <Duration> 150  </Duration> 
                    relevant_windows: [[0.96, 1]]'

                    query: <Query> A view of a bamboo fountain of water in a tea house and people scoop from and wash off <Query> 
                    duration:  <Duration> 150  </Duration> 
                    relevant_windows: [[0.21, 0.99]]'

                    query: <Query> The weather map shows snowfall <Query> 
                    duration:  <Duration> 150  </Duration> 
                    relevant_windows: [[0.12, 0.17],[0.27, 0.30],[0.32, 0.42],[0.43, 0.50],[0.68, 0.70],[0.80, 0.82]]'
                """


        format_text = """[[x, y],[a,b],[c,d]]
            if there is only one valid frame use [[x,y]]
            they represent the persentages of the video duration
            Ensure that the windows are in ascending order and do not overlap.
        """

        prompt = f"""
        Do not hallucinate \n
        follow the flowing text as accurate as possible \n

        Example: <Example> {example} </Example> \n
        Format: <Format> {format_text} </Format> \n
        Query: <Query> {query} </Query> 
        Duration: <Duration> {ann["duration"]} </Duration> \n

        For die video give me the relevant windows matching the Query for the given duration \n
        relevant_windows:  \n
        """

        text_input = prompt 

        query_prompt = "Query: " + query + "\n"
        task_prompt = "Given the video and the query, find the relevant windows.\nRelevant windows: "
        text_input = query_prompt + task_prompt

        out = {
            "text_input": text_input,
            "query_prompt": query_prompt,
            "task_prompt": task_prompt,
            "relevant_windows": str(ann["relevant_windows"]),
            "video": video,
            "timestamps": timestamps,
            "video_prompt_end": "<extra_id_0>",
            "duration": duration,
            "query_id": ann["qid"],
        }

        if self.audio_processor is not None:
            audio = self.audio_processor(video_path)
            out["audio"] = audio

        return out