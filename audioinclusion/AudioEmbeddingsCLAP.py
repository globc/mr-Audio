#from lavis.models.CLAP.examples.zero_shot_classification import audio_embeddings
import os

#from lavis.models.CLAP.msclap import CLAP
import torch
import ffmpeg
import numpy as np
import torchaudio
import io
import torchvision.io
from transformers import ClapProcessor, ClapModel
import torch
import torchaudio.transforms as T


class CLAPAudioEmbeddings:
    def __init__(self):
        self.eigendevice = torch.device('cpu' if (os.environ.get('USE_CPU_ONLY', '0') == '1')
                                        else 'cuda' if torch.cuda.is_available() else 'cpu')

        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused",
                                                    cache_dir=os.getcwd() + "/cache")
        self.clap_model.to(self.eigendevice)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused",
                                                    cache_dir=os.getcwd() + "/cache")
        #self.processor.to(self.eigendevice)

        #self.to(self.eigendevice)
        #if not (os.environ.get('USE_CPU_ONLY', '0') == '0'):
        #    self.clap_model.to("cuda")
        #    self.processor.to("cuda")


    #@staticmethod
    def get_audio_embeddings(self, audio_clips, sr=48000):
        #with torch.cuda.amp.autocast(enabled=(self.eigendevice != torch.device("cpu"))):
        #print(f"audio shape: {audio_clips.shape}")
        embeddings_lst = []
        for batch in audio_clips:
            audio_inputs = self.processor(audios=batch.cpu().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
            audio_inputs = audio_inputs.to(self.eigendevice)
            with torch.no_grad():
                audio_embeddings = self.clap_model.get_audio_features(**audio_inputs)
            embeddings_lst.append(audio_embeddings)
        embeddings = torch.stack(embeddings_lst, dim=0)
        #print(f"audio embdinngs shape: {embeddings.shape}")
        return embeddings

    #@staticmethod
    def read_audio(self, path_to_file, target_sr=48000):
        audio_waveform, sr = torchaudio.load(path_to_file)

        # Convert to mono if needed
        if audio_waveform.shape[0] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)

        # Resample to target sampling rate
        if sr != target_sr:
            audio_waveform = self.resample_audio(audio_waveform, sr, target_sr)
        audio_tensor = audio_waveform.squeeze(0)

        return audio_tensor, sr

    #@staticmethod
    def read_vid_with_audio(self, path_to_file, target_sr=48000, unit="sec"):
        video_frames, audio_waveform, info = torchvision.io.read_video(path_to_file, pts_unit=unit)

        # Convert to mono if needed
        if audio_waveform.shape[0] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)

        # Resample to target sampling rate
        if info["audio_fps"] != target_sr:
            audio_waveform = self.resample_audio(audio_waveform, info["audio_fps"], target_sr)
            info["audio_fps"] = target_sr
        audio_tensor = audio_waveform.squeeze(0)

        return video_frames, audio_tensor, info

    #@staticmethod
    def resample_audio(self,waveform, orig_sr, target_sr):
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        audio_waveform = resampler(waveform)  # Shape: [1, Resampled Samples]
        return audio_waveform
