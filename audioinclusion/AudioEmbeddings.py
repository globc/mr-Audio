#from lavis.models.CLAP.examples.zero_shot_classification import audio_embeddings
import os

from lavis.models.CLAP.msclap import CLAP
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
    def get_audio_embeddings(self, path_to_file, audio_name): #, audio):
        with torch.no_grad():
            #audio_time_series = self.read_audio(path_to_file, audio_name)#, audio)

            audio_path = [os.getcwd() + '/mr_BLIP_data/audio_files/' + audio_name + '.wav']
            audio_embeddings = self.clap_model.get_audio_embeddings(
                audio_files= audio_path
            )
        return audio_embeddings

    #def read_audio(self, audio_path, resample=True):
    #    torchaudio.set_audio_backend("ffmpeg")
    #    waveform, self.sample_rate = torchaudio.load(audio_path)
    #    if resample and self.sample_rate != 48000:
    #        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
    #        waveform = resampler(waveform)
    #        self.sample_rate = 48000  #  Update sample rate after resampling
    #    return waveform

    def read_audio(self, path_to_file, audio, resample=True):

        audio_samples = audio.to_soundarray(fps=48000)
        if audio_samples.ndim > 1:
            audio_samples = audio_samples.mean(axis=1)     # Convert stereo to mono if necessary

        waveform = torch.from_numpy(audio_samples).float()
        sample_rate = 48000

        audio.close()
        #video.close()

        return waveform.unsqueeze(0)