#from lavis.models.CLAP.examples.zero_shot_classification import audio_embeddings
import os

from lavis.models.CLAP.msclap import CLAP
from transformers import ClapProcessor, ClapModel
import torch
import ffmpeg
import numpy as np
import torchaudio
import io


class AudioEmbeddings:
    def __init__(self):
        self.clap_model = CLAP(
            version='2023',
            use_cuda=not (os.environ.get('USE_CPU_ONLY', '0') == '0')
        )


    #@classmethod         add cls
    def get_audio_embeddings(self, path_to_file): #, audio):
        with torch.no_grad():
            #audio_time_series = self.read_audio(path_to_file, audio_name)#, audio)

            processor = ClapProcessor.from_pretrained("openai/clap")
            model = ClapModel.from_pretrained("openai/clap")
        return audio_embeddings


    def read_audio(self, path_to_file, audio, resample=True):

        audio_samples = audio.to_soundarray(fps=48000)
        if audio_samples.ndim > 1:
            audio_samples = audio_samples.mean(axis=1)     # Convert stereo to mono if necessary

        waveform = torch.from_numpy(audio_samples).float()
        sample_rate = 48000

        audio.close()
        #video.close()

        return waveform.unsqueeze(0)