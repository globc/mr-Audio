#from lavis.models.CLAP.examples.zero_shot_classification import audio_embeddings
import os

from lavis.models.CLAP.msclap import CLAP
import torch


class AudioEmbeddings():
    def __init__(
            self,
            waveform,
            sample_rate = None #should be 48000
    ):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.clap_model = CLAP(version = '2023', use_cuda= not (os.environ.get('USE_CPU_ONLY', '0') == '0')) #.from_pretrained('clap_en_fusion')
        #self.clap_model.eval()

    #@classmethod         add cls
    def get_audio_embeddings(self, path_to_file):
        with torch.no_grad():
            audio_embeddings = self.clap_model.get_audio_embeddings(
                #self.waveform,
                #self.sample_rate,
                path_to_file
            )

            #audio_embeddings = audio_embeddings.cpu().numpy()
        return audio_embeddings