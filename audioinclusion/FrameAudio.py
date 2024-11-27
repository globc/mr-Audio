from moviepy.editor import VideoFileClip
from pyarrow import timestamp
from torch.utils.tensorboard.summary import video
import torchaudio
import cv2

class FrameAudio():
    def __init__(
            self,
            video_path,
            frame_index, #timestamp in seconds t


    ):
        self.video_path = video_path
        #self.frame_rate = cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FPS)
        self.frame_index = frame_index

    #@classmethod       add cls
    def get_audio_segment(self):

        video = VideoFileClip(self.video_path)
        frame_rate = video.fps
        timestamp_seconds = self.frame_index / frame_rate
        frame_duration = 1 / self.frame_rate


        audio = video.audio
        start_time = timestamp_seconds
        end_time = start_time + frame_duration
        audio_segment = audio.subc(start_time, end_time)

        return audio_segment


    def prepare_audio(self):
        waveform, sample_rate = torchaudio.load(self.video_path)
        if sample_rate != 48000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
            waveform = resampler(waveform)

        return waveform