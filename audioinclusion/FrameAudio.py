from moviepy.editor import VideoFileClip
from pyarrow import timestamp
from torch.utils.tensorboard.summary import video
import torchaudio
import cv2
import os

class FrameAudio():
    def __init__(
            self,
            video_path,
            vname,
            frame_index, #timestamp in seconds t
    ):
        self.video_path = video_path
        self.vname = vname
        #self.frame_rate = cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FPS)
        self.frame_index = frame_index[0] #TODO: how to use frame index correctly

    #@classmethod       add cls
    def get_audio_segment(self):
        """ Audio clip is taken from video and saved as wav"""


        video = VideoFileClip(self.video_path)
        frame_rate = video.fps
        timestamp_seconds = self.frame_index / frame_rate
        #TODO: check duration, make it longer
        frame_duration = 1 / frame_rate

        #Get Audio Data from the Clip
        audio = video.audio
        #start_time = timestamp_seconds
        start_time = self.frame_index
        start_time = int(start_time.cpu().item())
        end_time = start_time +1 #+ float(frame_duration) #+ 20
        audio_segment = audio.subclip(start_time, end_time)

        #Check if a File and Dir exists and save the audio clip was .wav if it does not exist yet
        base_dir = os.path.join(os.getcwd(), 'mr_BLIP_data')
        target_dir = os.path.join(base_dir, 'audio_files')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        output_path = os.path.join(target_dir, f"{self.vname}.wav")
        if not os.path.exists(output_path):
            audio_segment.write_audiofile(output_path, codec="pcm_s16le")

        return audio_segment, self.vname

    #@classmethod
    def prepare_audio(self):
        waveform, sample_rate = torchaudio.load(self.video_path)
        if sample_rate != 48000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
            waveform = resampler(waveform)

        return waveform