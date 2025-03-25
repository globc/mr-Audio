import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGatedCrossAttention(nn.Module):
    def __init__(self, vision_dim=768, audio_dim=512, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        # Projection layer to align feature dimensions
        self.audio_proj = nn.Linear(audio_dim, vision_dim)  # Project audio to 768

        # Cross-Attention (Audio -> Vision)
        self.audio_to_vision_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-Attention (Vision → Audio)
        self.vision_to_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Layer Norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Gating mechanism
        self.gate = nn.Linear(vision_dim, 1)  # Computes importance score per token
        # Feedforward Network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

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
        audio_emb = self.audio_proj(audio_emb)  # Shape: (160, 32, 768)

        # 1. Audio-Guided Vision Attention (Audio as Query, Vision as Key/Value)
        attn_audio_to_vision, _ = self.audio_to_vision_attn(audio_emb, vision_emb, vision_emb)

        # 2. Vision-Guided Audio Attention (Vision as Query, Audio as Key/Value)
        attn_vision_to_audio, _ = self.vision_to_audio_attn(vision_emb, audio_emb, audio_emb)

        # 3. Compute Gated Fusion Weights
        modality_scores = torch.softmax(self.gate(audio_emb.mean(dim=1)), dim=-1)  # Shape [batch, 2]

        # Expand modality scores to match query_tokens (dim=1)
        modality_scores = modality_scores.unsqueeze(1).expand(-1, audio_emb.shape[1], -1)

        # 4. Weighted Sum of Both Attention Outputs
        fused_representation = (
                modality_scores[:, 0].unsqueeze(1) * attn_audio_to_vision +
                modality_scores[:, 1].unsqueeze(1) * attn_vision_to_audio
        )

        # 5. Apply Residual Connection & Normalization
        fused_representation = self.norm1(fused_representation + audio_emb + vision_emb)

        # 6. Feedforward Network
        ff_output = self.ff(fused_representation)
        fused_representation = self.norm2(ff_output + fused_representation)

        return fused_representation  # Shape: (160, 32, 768)
