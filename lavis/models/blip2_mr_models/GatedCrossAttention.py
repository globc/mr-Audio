import torch
import torch.nn as nn
from sympy import false


class GatedCrossAttention(nn.Module):
    def __init__(self, vision_dim=768, audio_dim=512, hidden_dim=768, num_heads=8):
        super().__init__()

        # Projection layer to align feature dimensions
        self.audio_proj = nn.Linear(audio_dim, vision_dim)  # Project audio to 768

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Gating mechanism
        self.gate = nn.Linear(hidden_dim, 1)  # Computes importance score per token

    def forward(self, vision_emb, audio_emb):
        """
        Args:
            vision_emb (torch.Tensor): A tensor of shape [batch_size, seq_len, vision_dim] representing vision embeddings.
            audio_emb (torch.Tensor):  A tensor of shape [batch_size, seq_len, audio_dim] representing audio embeddings.
        
        Returns:
            torch.Tensor: A tensor of shape [batch_size, seq_len, hidden_dim] representing the fused representation 
                          of vision and audio embeddings.
        """
        # Project audio features to match vision feature size
        if audio_emb.shape[-1] != vision_emb.shape[-1]:
            audio_emb = self.audio_proj(audio_emb)

        # Compute cross-attention (audio attending to video)
        attended_audio, _ = self.cross_attn(audio_emb, vision_emb, vision_emb)

        # Compute gating scores (importance of each token)
        gate_values = torch.sigmoid(self.gate(audio_emb))

        fused_representation = gate_values * attended_audio + (1 - gate_values) * vision_emb

        return fused_representation
