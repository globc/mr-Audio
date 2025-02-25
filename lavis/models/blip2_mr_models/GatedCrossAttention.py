import torch
import torch.nn as nn

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
        vision_emb: [batch_size, seq_len, vision_dim]  -> (160, 32, 768)
        audio_emb:  [batch_size, seq_len, audio_dim]   -> (160, 32, 512)
        """
        # Project audio features to match vision feature size
        #audio_emb = self.audio_proj(audio_emb)  # Shape: (160, 32, 768)
        # comment in line in for fusion before projection

        # Compute cross-attention (audio attending to video)
        attended_audio, _ = self.cross_attn(audio_emb, vision_emb, vision_emb)  # Shape: (160, 32, 768)

        # Compute gating scores (importance of each token)
        gate_values = torch.sigmoid(self.gate(audio_emb))  # Shape: (160, 32, 1)

        # Apply gating mechanism
        fused_representation = gate_values * attended_audio + (1 - gate_values) * vision_emb  # Selectively fuse

        return fused_representation  # Shape: (160, 32, 768)
