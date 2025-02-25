import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, audio_dim=512, visual_dim=768, hidden_dim=256, num_heads=8, dropout=0.1, output_dim=768,use_visual_enc=False):
        super(MultimodalTransformer, self).__init__()

        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_visual_enc = use_visual_enc

        # Input Embeddings (Linear projections to a common hidden dimension)
        self.audio_embedding = nn.Linear(audio_dim, hidden_dim)
        self.visual_embedding = nn.Linear(visual_dim, hidden_dim)

        # Modality-Specific Transformers (optional, but often beneficial)
        self.audio_transformer = TransformerEncoderBlock(hidden_dim, num_heads, dropout)
        self.visual_transformer = TransformerEncoderBlock(hidden_dim, num_heads, dropout)

        # Cross-Modal Attention (the core of the fusion)
        self.cross_attention = CrossAttentionBlock(hidden_dim, num_heads, dropout)

        # Fusion Layer
        self.fusion_layer = TransformerEncoderBlock(hidden_dim, num_heads, dropout) # or MLP

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, audio_features, visual_features):
        # 1. Input Embeddings
        if audio_features.shape[-1] != self.hidden_dim:
            audio_embedded = self.audio_embedding(audio_features)
        else:
            audio_embedded = audio_features
        if visual_features.shape[-1] != self.hidden_dim:
            visual_embedded = self.visual_embedding(visual_features)
        else:
            visual_embedded = visual_features

        # 2. Modality-Specific Transformers
        audio_transformed = self.audio_transformer(audio_embedded)
        if self.use_visual_enc:
            visual_transformed = self.visual_transformer(visual_embedded)
        else:
            visual_transformed = visual_features

        # 3. Cross-Modal Attention
        fused_representation = self.cross_attention(audio_transformed, visual_transformed)

        # 4. Fusion Layer
        fused_representation = self.fusion_layer(fused_representation)

        # 5. Output Layer
        output = self.output_layer(fused_representation)
        return output


class TransformerEncoderBlock(nn.Module):  # Standard Transformer Encoder Block
    def __init__(self, hidden_dim, num_heads, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),  # Example Feed-Forward
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pe = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, x):
        x = self.pe(x)
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class CrossAttentionBlock(nn.Module): # Modified Transformer Block for Cross-Attention
    def __init__(self, hidden_dim, num_heads, dropout):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential( # Same FF as before
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):  # x and y are the two modalities
        attn_output, _ = self.cross_attention(x, y, y)  # Query x, Key and Value y
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim

        # Compute sinusoidal encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]
