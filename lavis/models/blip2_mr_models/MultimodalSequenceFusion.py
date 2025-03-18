import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import logging

class MultimodalSequenceFusion(nn.Module):
    def __init__(
            self,
            embed_dim_audio=512, # 512 for CLAP
            embed_dim_image=768, # 768 for CLIP
            n_heads=8,
            mode='stack_fusion'  # or 'stack_fusion', 'x-attention
    ):
        super().__init__()
        self.debug = False
        self.plot = False #Not Tested!
        self.embed_dim = embed_dim_audio
        self.n_heads = n_heads
        self.mode = mode

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim_audio,
            num_heads=n_heads,
            batch_first=True #[B, seq_len, embed_dim]
        )

        # Projection if cat_fusion (2*embed_dim -> embed_dim).
        self.cat_proj = nn.Linear(2 * embed_dim_audio, embed_dim_audio)

        self.out_proj = nn.Linear(embed_dim_audio, embed_dim_image)
        self.layernorm = nn.LayerNorm(embed_dim_audio)
        self.dropout = nn.Dropout(p=0.1)


        if self.mode == 'stack_fusion_token':  #idea from CLS token in BERT
            self.numtokens = 32
            self.fusion_tokens = nn.Parameter(torch.randn(1, self.numtokens, embed_dim_audio))

        #add learnable weights for each modality for stack fusion -> Weighted Sum
        if self.mode == 'stack_fusion':
            self.additional_weights = nn.Linear(embed_dim_audio, 1)

        if self.mode == 'weighted_sum_No_Paramsharing':
            self.additional_weights_audio = nn.Linear(embed_dim_audio, 1)
            self.additional_weights_image = nn.Linear(embed_dim_image, 1)

        if self.mode == 'audio_up_proj': #do fusion in image space
            self.image_space_linear = nn.Linear(embed_dim_image, embed_dim_image)
            self.additional_weights = nn.Linear(embed_dim_image, 1)



        # just use MLP for Fusion
        if self.mode == 'mlp_fusion':
            self.mlp = nn.Sequential(
                nn.Linear(2 * embed_dim_audio, 2 * embed_dim_audio),
                nn.ReLU(),
                nn.Linear(2 * embed_dim_audio, embed_dim_image)
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
            attn_out, attn_weights = self.mha(
                query=fusion_tokens,
                key=combined_seq,
                value=combined_seq
            )
            out = self.dropout(attn_out) + fusion_tokens
            out = self.layernorm(out)
            fused_output = self.out_proj(out)#.squeeze(1) #
            return fused_output, attn_weights #problem with shapes afterwards since 33 tokens


        elif self.mode == 'stack_fusion': #This is Multimodal Sequence Fusion
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------", flush=True)
                print("We are in Stack Fusion", flush=True)
            # Token-level concatenation along seq_len dimension
            # Result: shape [B, (A+I), embed_dim], Example [B, 64, 512]
            # Then standard self-attention over all tokens
            combined_seq = torch.cat([image_emb, audio_emb], dim=1)


            attn_out, attn_weights = self.mha(
                query=combined_seq,
                key=combined_seq,
                value=combined_seq
            )

            #out = self.dropout(attn_out) + combined_seq
            out = attn_out + combined_seq
            out = self.layernorm(out)

            B, S, E = out.shape
            out = out.view(B, 2, S // 2, E)

            #out = out.mean(dim=1)
            weights = self.additional_weights(out)
            weights = F.softmax(weights, dim=1)
            weighted_sum = (out * weights).sum(dim=1)

            fused_output = self.out_proj(weighted_sum)

            return fused_output, attn_weights #torch.Size([160, 32, 768]), something

        elif self.mode == 'weighted_sum_No_Paramsharing': #This is Multimodal Sequence Fusion
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------", flush=True)
                print("We are in weighted_sum_No_Paramsharing", flush=True)
            # Token-level concatenation along seq_len dimension
            # Result: shape [B, (A+I), embed_dim], Example [B, 64, 512]
            # Then standard self-attention over all tokens
            combined_seq = torch.cat([image_emb, audio_emb], dim=1)


            attn_out, attn_weights = self.mha(
                query=combined_seq,
                key=combined_seq,
                value=combined_seq
            )

            #out = self.dropout(attn_out) + combined_seq
            out = attn_out + combined_seq
            out = self.layernorm(out)

            B, S, E = out.shape
            out = out.view(B, 2, S // 2, E)
            audio = out[:, 0, :, :]
            image = out[:, 1, :, :]
            #out = out.mean(dim=1)
            weights_audio = self.additional_weights_audio(audio)
            weights_image = self.additional_weights_image(image)

            weights_audio = F.softmax(weights_audio, dim=1)
            weights_image = F.softmax(weights_image, dim=1)
            weights = torch.stack([weights_audio, weights_image], dim=1)

            weighted_sum = (out * weights).sum(dim=1)

            fused_output = self.out_proj(weighted_sum)

            return fused_output, attn_weights #torch.Size([160, 32, 768]), something

        elif self.mode == 'audio_up_proj': #do fusion in image space
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------", flush=True)
                print("We are in Stack Fusion", flush=True)
            # Token-level concatenation along seq_len dimension
            # Result: shape [B, (A+I), embed_dim], Example [B, 64, 512]
            # Then standard self-attention over all tokens
            combined_seq = torch.cat([image_emb, audio_emb], dim=1)


            attn_out, attn_weights = self.mha(
                query=combined_seq,
                key=combined_seq,
                value=combined_seq
            )

            #out = self.dropout(attn_out) + combined_seq
            out = attn_out + combined_seq
            out = self.layernorm(out)

            B, S, E = out.shape
            out = out.view(B, 2, S // 2, E)

            #out = out.mean(dim=1)
            weights = self.additional_weights(out)
            weights = F.softmax(weights, dim=1)
            weighted_sum = (out * weights).sum(dim=1)

            fused_output = self.image_space_linear(weighted_sum)

            return fused_output, attn_weights #torch.Size([160, 32, 768]), something

        elif self.mode == 'weighted_sum_only': #TODO Try 3
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------", flush=True)
                print("We are in weighted_sum_only", flush=True)
            # Token-level concatenation along seq_len dimension
            # Result: shape [B, (A+I), embed_dim], Example [B, 64, 512]
            # Then standard self-attention over all tokens
            combined_seq = torch.cat([image_emb, audio_emb], dim=1)

            B, S, E = combined_seq.shape
            out = combined_seq.view(B, 2, S // 2, E)

            #out = out.mean(dim=1)
            weights = self.additional_weights(out)
            weights = F.softmax(weights, dim=1)
            weighted_sum = (out * weights).sum(dim=1)

            fused_output = self.out_proj(weighted_sum)

            return fused_output, None #torch.Size([160, 32, 768]), something

        elif self.mode == 'cat_fusion':
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in Cat Fusion")

            if audio_emb.shape[1] != image_emb.shape[1]:
                raise ValueError(
                    f"For cat_fusion, seq_len must match. "
                    f"Got audio {audio_emb.shape[1]}, image {image_emb.shape[1]}"
                )

            combined_seq = torch.cat([image_emb, audio_emb], dim=-1)  # [B, seq_len, 2*embed_dim]

            combined_seq_proj = self.cat_proj(combined_seq)  # [B, seq_len, embed_dim]

            attn_out, attn_weights = self.mha( # attention here doesnt make sense since the embeddings are all over the place
                query=combined_seq_proj,
                key=combined_seq_proj,
                value=combined_seq_proj
            )


            out = self.dropout(attn_out) + combined_seq_proj
            out = self.layernorm(out)

            fused_output = self.out_proj(out)

            return fused_output, attn_weights

        elif self.mode == 'x-attention': # TODO Try 2 Take audio as query
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in X Fusion")


            attn_out, attn_weights = self.mha(
                query=image_emb,  # [B, I, 512]
                key=audio_emb,  # [B, A, 512]
                value=audio_emb
            )

            out = self.dropout(attn_out) + image_emb
            out = self.layernorm(out)

            fused_output = self.out_proj(out)

            return fused_output, attn_weights

        elif self.mode == 'mlp_fusion': #TODO Try 1
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in MLP Fusion")
            combined_seq = torch.cat([image_emb, audio_emb], dim=-1) #  [B, tokens, 2* 512], [B, seq_len, 2*embed_dim]
            mlp_out = self.mlp(combined_seq)
            #fused_output = self.out_proj(mlp_out)
            return mlp_out, None

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Choose from ['stack_fusion', 'cat_fusion', 'x-attention', 'stack_fusion_token', 'mlp_fusion']."
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