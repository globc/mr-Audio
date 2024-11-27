#from lavis.models.CLAP.examples.zero_shot_classification import audio_embeddings
from lavis.models.CLAP.msclap.models.clap import CLAP
import torch


class AudioEmbeddings():
    def __init__(
            self,
            waveform,
            sample_rate = None #should be 48000
    ):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.clap_model = CLAP.from_pretrained('clap_en_fusion')
        self.clap_model.eval()

    #@classmethod         add cls
    def get_audio_embeddings(self):
        with torch.no_grad():
            audio_embeddings = self.clap_model.get_audio_embeddings_from_waveform(self.waveform, self.sample_rate)

            #audio_embeddings = audio_embeddings.cpu().numpy()
        return audio_embeddings