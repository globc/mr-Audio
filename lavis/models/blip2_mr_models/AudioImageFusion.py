import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import logging

class AudioImageFusion(nn.Module):
    def __init__(
            self,
            embed_dim=512, # 512 for CLAP
            n_heads=8,
            mode='stack_fusion'  # or 'stack_fusion', 'x-attention
    ):
        super().__init__()

        self.debug = False
        self.plot = False #Not Tested!

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.mode = mode

        # Multihead attention (PyTorch: query, key, value)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True #[B, seq_len, embed_dim]
        )

        # Projection if cat_fusion (2*embed_dim -> embed_dim).
        self.cat_proj = nn.Linear(2 * embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, 768) #2048 if directly into T5, or 768 if into T5 Proj LAyer (because T5 Proj layer is trained)

        self.layernorm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, audio_emb, image_emb):

        if self.debug:
            logging.info(f"audio_emb type: {type(audio_emb)}")
            logging.info(f"image_emb type: {type(image_emb)}")




        # For batch_first: [batch_size, seq_len, embed_dim].
        if audio_emb.ndim == 2:
            audio_emb = audio_emb.unsqueeze(0)# -> [1, 32, 512]

            #if self.debug:
                #print(f"[DEBUG] Reshaped audio_emb to: {audio_emb.shape}")
        if image_emb.ndim == 2:
            image_emb = image_emb.unsqueeze(0) # -> [1, 32, 512] example

            #if self.debug:
                #print(f"[DEBUG] Reshaped image_emb to: {image_emb.shape}")
        #  image_emb is [1, 32, 512], audio_emb is [1, 32, 512]

        if self.mode == 'stack_fusion':
            # Token-level concatenation along seq_len dimension
            # Result: shape [B, (A+I), embed_dim], Example [B, 64, 512]
            # Then standard self-attention over all tokens
            combined_seq = torch.cat([image_emb, audio_emb], dim=1)

            #if self.debug:
                #print(f"[DEBUG] combined_seq (stack) shape: {combined_seq.shape}")

            # Self-attention: Q=K=V= combined_seq
            attn_out, attn_weights = self.mha(
                query=combined_seq,
                key=combined_seq,
                value=combined_seq
            )
            # attn_out => [B, (A+I), embed_dim]
            # Residual & projection
            # ((typical Transformer block is attn_out + input, then norm))
            out = self.dropout(attn_out) + combined_seq
            out = self.layernorm(out)


            B, S, E = out.shape
            out = out.view(B, 2, S//2, E)
            out = out.mean(dim=1)
            fused_output = self.out_proj(out)


            return fused_output, attn_weights

        elif self.mode == 'cat_fusion': #TODO: Test this mode
            # Feature-level concatenation along the embedding dimension
            # shape [B, seq_len, 2*embed_dim]
            # Then project down to [B, seq_len, embed_dim]
            #do self-attention on that (still seq_len = 32).
            # audio_emb and image_emb have the same seq_len (e.g., 32).
            if audio_emb.shape[1] != image_emb.shape[1]:
                raise ValueError(
                    f"For cat_fusion, seq_len must match. "
                    f"Got audio {audio_emb.shape[1]}, image {image_emb.shape[1]}"
                )

            combined_seq = torch.cat([image_emb, audio_emb], dim=-1)  # [B, seq_len, 2*embed_dim]

            if self.debug:
                print(f"[DEBUG] combined_seq (cat) shape: {combined_seq.shape}")

            combined_seq_proj = self.cat_proj(combined_seq)  # [B, seq_len, embed_dim]

            if self.debug:
                print(f"[DEBUG] combined_seq_proj shape: {combined_seq_proj.shape}")

            # Self-attention on combined_seq_proj
            attn_out, attn_weights = self.mha(
                query=combined_seq_proj,
                key=combined_seq_proj,
                value=combined_seq_proj
            )
            # attn_out => [B, seq_len, embed_dim]


            out = self.dropout(attn_out) + combined_seq_proj
            out = self.layernorm(out)

            B, S, E = out.shape
            out = out.view(B, 2, S // 2, E)
            out = out.mean(dim=-1)
            fused_output = self.out_proj(out)
            print(f"[DEBUG] fused_output shape: {fused_output.shape}")

            return fused_output, attn_weights

        elif self.mode == 'x-attention': #TODO: Test this mode
            # Cross-attention: we use image as Query, audio as Key/Value
            # shapes: Q => [B, I, embed_dim], K=> [B, A, embed_dim], V=> [B, A, embed_dim]

            if self.debug:
                print("--------------------------------------------------------------------------------------------------")
                print(f"[DEBUG] x-attention => Q=image, K/V=audio")

            attn_out, attn_weights = self.mha(
                query=image_emb,  # [B, I, 512]
                key=audio_emb,  # [B, A, 512]
                value=audio_emb
            )
            # attn_out => [B, I, embed_dim]

            out = self.dropout(attn_out) + image_emb
            out = self.layernorm(out)

            fused_output = self.out_proj(out)
            if self.plot:
                self.visualize_attention_wandb(attn_weights)

            return fused_output, attn_weights

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Choose from ['stack_fusion', 'cat_fusion', 'x-attention']."
            )

    def visualize_attention_wandb(self, attn_weights, head=0, title="Attention Map", step=None, log_key="Attention Map"):
        """
        Visualize and log an attention map to wandb.

        Args:
            attn_weights (torch.Tensor): Tensor of shape [B, n_heads, query_len, key_len].
            head (int): The head index to visualize (default: 0).
            title (str): Title for the plot.
            step (int or None): Training step for logging.
            log_key (str): Key under which the image will be logged in wandb.

        Returns:
            None
        """
        # Use the first batch element for visualization.
        attn = attn_weights[0, head].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(attn, cmap="viridis", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")

        # Log the figure as an image to wandb.
        wandb.log({log_key: wandb.Image(fig)}, step=step)
        plt.close(fig)