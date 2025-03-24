import torch
import torch.nn as nn

class ImageAudioFusion(nn.Module):
    """Experimental Fusion Method. This class was not used to produce the results in the paper.
    This module fuses image and audio features using an attention mechanism.
    Fusion happens at audio embedding dimension.
    Args:
        embed_dim: embedding dimension
        num_heads: number of attention

    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)


    def forward(self, concat_vector):
        """
        image_feats: (batch_size,embed_dim) e.g. [32,512] (latter is audio encoder output len
        audio_feats: (batch_size,embed_dim) e.g. [32,512]
        concat vector is both vectors concatenated
        """

        attn_output, _ = self.attn(query=concat_vector, key=concat_vector, value=concat_vector)
        # shape is (2,B,E), take mean or pool token
        #here, mean is taken across two tokens
        # shape => (B, E)

        fused = attn_output.mean(dim=0)
        fused = self.linar(fused)
        return fused