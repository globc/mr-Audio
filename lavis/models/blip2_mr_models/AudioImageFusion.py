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

        #Learnable fusion token TODO Try 1
        if self.mode == 'stack_fusion_token':  #idea from CLS token in BERT
            self.numtokens = 32
            self.fusion_tokens = nn.Parameter(torch.randn(1, self.numtokens, embed_dim))

        #add learnable weights for each modality for stack fusion TODO Try 2
        if self.mode == 'stack_fusion':
            self.additional_weights = nn.Linear(embed_dim, 1)


        #define stuff for second mha:
        #TODO Try 3
        self.mha2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True #[B, seq_len, embed_dim]
        )

        self.dropout2 = nn.Dropout(p=0.1)
        self.layernorm2 = nn.LayerNorm(embed_dim)



        #TODO Try 4
        # just use MLP for Fusion
        if self.mode == 'mlp_fusion':
            self.mlp = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )


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

        if self.mode == 'stack_fusion_token':
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in Stack Fusion Token")
            """Here we basically add fusion tokens to the other tokens, that is learned during training"""

            combined_seq = torch.cat([image_emb, audio_emb[:,:1,:]], dim=1) #  [1, 64, 512], [B, seq_len, embed_dim], or 33 instead of 64
            B = combined_seq.shape[0]
            fusion_tokens = self.fusion_tokens.expand(B, -1, -1) # e.g. [1, 1, 512], [B, 1, embed_dim]
            #combined_seq = torch.cat([fusion_tokens, combined_seq], dim=1) # [1, 65, 512], [B, seq_len+1, embed_dim]

            attn_out, attn_weights = self.mha(
                query=fusion_tokens,
                key=combined_seq,
                value=combined_seq
            )
            out = self.dropout(attn_out) + fusion_tokens
            out = self.layernorm(out)

            #fusion_out = out[:, 0:1, :] #should be[B, 1, embed_dim] torch.Size([160, 1, 512])

            #print("fused_output shape Stack Fusion Token: ", fusion_out.shape)
            fused_output = self.out_proj(out)#.squeeze(1) #

            #print("-------------------------------------------------------------------------------------------------------")
            #print(f"[DEBUG] fused_output shape Stack Fusion Token: {fused_output.shape}") #
            return fused_output, attn_weights


        elif self.mode == 'stack_fusion':
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------", flush=True)
                print("We are in Stack Fusion", flush=True)
            # Token-level concatenation along seq_len dimension
            # Result: shape [B, (A+I), embed_dim], Example [B, 64, 512]
            # Then standard self-attention over all tokens
            combined_seq = torch.cat([image_emb, audio_emb], dim=1) #select only one audio embedding


            attn_out, attn_weights = self.mha(
                query=combined_seq,
                key=combined_seq,
                value=combined_seq
            )

            out = self.dropout(attn_out) + combined_seq
            out = self.layernorm(out)

            #TODO Try 3
            #attn_out2, attn_weights2 = self.mha2(
            #    query=out,
            #    key=out,
            #    value=out
            #)
            #out = self.dropout2(attn_out2) + out
            #out = self.layernorm2(out)


            B, S, E = out.shape
            out = out.view(B, 2, S // 2, E)

            #out = out.mean(dim=1)
            weights = self.additional_weights(out)
            weights = F.softmax(weights, dim=1)
            weighted_sum = (out * weights).sum(dim=1)

            #token_weights = self.additional_weights(out)
            #token_weights = F.softmax(token_weights, dim=1)
            #weighted_embedding = (out * token_weights).sum(dim=1) #weighted sum over tokens

            #modalities = torch.stack([image_emb, audio_emb], dim=2)  # shape: [B, 32, 2, embedding_dim], need to take audio embedding copied 32 times because of reshaping functions down the pipeline
            #B, T, M, E = modalities.shape # B=160, T=32, M=2, E=512, M is modalities
            #modalities_flat = modalities.view(B * T * M, E) # shape: [B*32*2, embedding_dim]
            #modality_logits = self.additional_weights(modalities_flat) # shape: [B*32*2, 1]
            #modality_logits = modality_logits.view(B, T, M, 1)  # shape: [B, 32, 2, 1]
            #modality_weights = F.softmax(modality_logits, dim=2)  # shape: [B, 32, 2, 1]
            #fused_tokens = (modalities * modality_weights).sum(dim=2)  # shape: [B, 32, embedding_dim]




            #out = out.mean(dim=1) #ORIGINAL old method to average the two modalities


            fused_output = self.out_proj(weighted_sum)

            return fused_output, attn_weights #torch.Size([160, 32, 768]), something

        elif self.mode == 'cat_fusion':
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in Cat Fusion")
            # Feature-level (embedding dim) concatenation along the embedding dimension
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

            combined_seq_proj = self.cat_proj(combined_seq)  # [B, seq_len, embed_dim]

            # Self-attention on combined_seq_proj
            attn_out, attn_weights = self.mha(
                query=combined_seq_proj,
                key=combined_seq_proj,
                value=combined_seq_proj
            )
            # attn_out => [B, seq_len, embed_dim]


            out = self.dropout(attn_out) + combined_seq_proj
            out = self.layernorm(out)

            #print(f"[DEBUG] out shape: {out.shape}") #out shape: torch.Size([160, 32, 512])
            fused_output = self.out_proj(out)
            #print("--------------------------------------------------------------------------------------------------")
            #print(f"[DEBUG] fused_output shape Cat Attn: {fused_output.shape}") #torch.Size([160, 32, 768])

            return fused_output, attn_weights

        elif self.mode == 'x-attention':
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in X Fusion")

            # Cross-attention: image as Query, audio as Key/Value
            # shapes: Q => [B, I, embed_dim], K=> [B, A, embed_dim], V=> [B, A, embed_dim]



            attn_out, attn_weights = self.mha(
                query=image_emb,  # [B, I, 512]
                key=audio_emb,  # [B, A, 512]
                value=audio_emb
            )
            # attn_out => [B, I, embed_dim]

            out = self.dropout(attn_out) + image_emb
            out = self.layernorm(out)
            #print("--------------------------------------------------------------------------------------------------")
            #print("We are in X Fusion")
            #print(f"[DEBUG] out shape: {out.shape}") #torch.Size([160, 32, 512])

            fused_output = self.out_proj(out)
            #print("--------------------------------------------------------------------------------------------------")
            #print(f"[DEBUG] fused_output shape X Attn: {fused_output.shape}") #torch.Size([160, 32, 768])

            return fused_output, attn_weights

        elif self.mode == 'mlp_fusion':
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in MLP Fusion")
            combined_seq = torch.cat([image_emb, audio_emb], dim=-1) #  [B, tokens, 2* 512], [B, seq_len, 2*embed_dim]
            mlp_out = self.mlp(combined_seq)
            fused_output = self.out_proj(mlp_out)
            return fused_output, None

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Choose from ['stack_fusion', 'cat_fusion', 'x-attention', 'stack_fusion_token']."
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