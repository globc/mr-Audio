"""
MultimodalSequenceFusion.py
This file contains the class MultimodalSequenceFusion (MSF), which fuses audio and image embeddings using self-attention and
subsequent explicit modularity weight-learning.
Fusion happens at audio embedding dimension level, image embeddings are projected to audio embedding dimension before
being passed as argument to MSF.
MSF creates a multimodal sequence of image and audio embedding sequences, applies self-attention and learns additional
weights for each modularity by applying a linear layer and using a Softmax activation function subsequently.

Additional Modes apart from MSF are available, but are experimental and do not perform as well as MSF for Moment Retrieval.

It also includes the following modes, which differ to MSF in the following:
    - 'weighted_sum_No_Paramsharing' -> MSF with separate fusion-weights for audio and image embeddings, uses Sigmoid gate
    - 'audio_up_proj' -> MSF with fusion at image space
    - 'stack_fusion_token' -> Learning an additional token for fusion, similar to CLS token in BERT but not competitive
    - 'cat_fusion' -> Stacking Audio and Image Embedding sequence on top of each other (along embedding dimension) and projecting into a common space, before applying self-attention
    - 'x-attention' -> Cross-attention between audio and image embeddings
    - 'mlp_fusion' -> Fusion using only a MLP
    - 'weighted_sum_only' -> Only weighted sum of audio and image embeddings, no self-attention

Dependencies:
    torch, torch.nn, torch.nn.functional, matplotlib, seaborn, wandb, logging

Usage Example:
    fusion_module = MultimodalSequenceFusion(mode='multimodal_sequence_fusion')
    fused_output, attn_weights = fusion_module(audio_emb, image_emb)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import logging

class AudioImageFusionModule(nn.Module):
    """
        MultimodalSequenceFusion Module
        This module fuses audio and image embeddings using self-attention and subsequent explicit modularity weight-learning.
        Additional modes besides Multimodal Sequence Fusion are available, though they are experimental and do not perform as well as MSF.

Parameters:
        embed_dim_audio (int): Dimensionality of the audio embeddings. Default is 512.
        embed_dim_image (int): Dimensionality of the image embeddings. Default is 768.
        n_heads (int): Number of attention heads to use in the multihead attention mechanism. Default is 8.
        mode (str): Fusion strategy to use. Supported modes include:
            - 'multimodal_sequence_fusion'
            - 'weighted_sum_No_Paramsharing'
            - 'audio_up_proj'
            - 'stack_fusion_token'
            - 'cat_fusion'
            - 'x-attention'
            - 'mlp_fusion'

    """

    def __init__(
            self,
            embed_dim_audio: int, # 512 for CLAP
            embed_dim_image: int, # 768 for CLIP
            n_heads=8,
            mode='multimodal_sequence_fusion'  # Experimental Modes: ['multimodal_sequence_fusion', 'cat_fusion', 'x-attention', 'stack_fusion_token', 'mlp_fusion']
    ):
        super().__init__()
        self.debug = False

        self.embed_dim = embed_dim_audio
        self.n_heads = n_heads
        self.mode = mode

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim_audio,
            num_heads=n_heads,
            batch_first=True #[B, seq_len, embed_dim]
        )

        self.out_proj = nn.Linear(embed_dim_audio, embed_dim_image)
        self.layernorm = nn.LayerNorm(embed_dim_audio)
        self.dropout = nn.Dropout(p=0.1)


        if self.mode == 'multimodal_sequence_fusion':
            self.additional_weights = nn.Linear(embed_dim_audio, 1)

        if self.mode == 'weighted_sum_No_Paramsharing':
            self.additional_weights_audio = nn.Linear(embed_dim_audio, 1)
            self.additional_weights_image = nn.Linear(embed_dim_audio, 1) #fusion at audio embedding level

        if self.mode == 'audio_up_proj': #do fusion in image space
            self.image_space_linear = nn.Linear(embed_dim_image, embed_dim_image)
            self.additional_weights = nn.Linear(embed_dim_image, 1)

        if self.mode == 'stack_fusion_token':  #idea from CLS token in BERT
            self.numtokens = 32
            self.fusion_tokens = nn.Parameter(torch.randn(1, self.numtokens, embed_dim_audio))

        if self.mode == 'cat_fusion':
            # Projection if cat_fusion (2*embed_dim -> embed_dim).
            self.cat_proj = nn.Linear(2 * embed_dim_audio, embed_dim_audio)

        # just use MLP for Fusion
        if self.mode == 'mlp_fusion':
            self.mlp = nn.Sequential(
                nn.Linear(2 * embed_dim_audio, 2 * embed_dim_audio),
                nn.ReLU(),
                nn.Linear(2 * embed_dim_audio, embed_dim_image)
            )


    def forward(self, audio_emb, image_emb):
        """
        Forward pass of the MultimodalSequenceFusion Module, performs fusion based on configured mode.
        Reshaping is applied if necessary. If debug mode active, prints shape information.
        input:
            audio_emb (torch.Tensor): Audio embeddings of shape [B, seq_len, embed_dim].
            image_emb (torch.Tensor): Image embeddings of shape [B, seq_len, embed_dim].
        returns:
            fused_output (torch.Tensor): Fused embeddings of shape [B, seq_len, embed_dim].
            attn_weights (torch.Tensor): Attention weights of shape [B, n_heads, seq_len, seq_len].
        """

        if self.debug:
            logging.info(f"audio_emb type: {type(audio_emb)}")
            logging.info(f"image_emb type: {type(image_emb)}")

        # For batch_first: [batch_size, seq_len, embed_dim].
        if audio_emb.ndim == 2:
            audio_emb = audio_emb.unsqueeze(0)# -> [1, 32, 512]
        if image_emb.ndim == 2:
            image_emb = image_emb.unsqueeze(0) # -> [1, 32, 512] example

        if self.mode == 'multimodal_sequence_fusion':
            if self.debug:
                logging.info("-------------------------------------------------------------------------------------------------------")
                logging.info("We are in Multimodal Sequence Fusion")
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


        elif self.mode == 'stack_fusion_token': #TODO: check if this is really learned token
            if self.debug:
                logging.info("-------------------------------------------------------------------------------------------------------")
                logging.info("We are in Stack Fusion Token")

            combined_seq = torch.cat([image_emb, audio_emb[:,:1,:]], dim=1) #  [1, 64, 512], [B, seq_len, embed_dim], here [1, 33, 512], 32 image + 1 audio
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



        elif self.mode == 'weighted_sum_No_Paramsharing':
            if self.debug:
                logging.info("-------------------------------------------------------------------------------------------------------")
                logging.info("We are in weighted_sum_No_Paramsharing")

            combined_seq = torch.cat([image_emb, audio_emb], dim=1)

            attn_out, attn_weights = self.mha(
                query=combined_seq,
                key=combined_seq,
                value=combined_seq
            )

            #out = self.dropout(attn_out) + combined_seq, Dropout optional
            out = attn_out + combined_seq
            out = self.layernorm(out)

            B, S, E = out.shape
            out = out.view(B, 2, S // 2, E)
            audio = out[:, 0, :, :]
            image = out[:, 1, :, :]

            #Seperate Weights are learned for audio and image embeddings
            weights_audio = self.additional_weights_audio(audio)
            weights_image = self.additional_weights_image(image)
            print(f"weights_audio: {weights_audio.shape}, weights_image: {weights_image.shape}") #TODO


            weights = torch.stack([weights_audio, weights_image], dim=1)
            print(f"weights: {weights.shape}") #TODO

            weighted_sum = (out * weights).sum(dim=1)

            fused_output = self.out_proj(weighted_sum)

            return fused_output, attn_weights #torch.Size([160, 32, 768]), attn_weights

        elif self.mode == 'audio_up_proj': #do fusion in image space
            if self.debug:
                logging.info("-------------------------------------------------------------------------------------------------------")
                logging.info("We are in Audio Up Proj")

            combined_seq = torch.cat([image_emb, audio_emb], dim=1) #Both sequences exist in audio embedding space

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

            weights = self.additional_weights(out)
            weights = F.softmax(weights, dim=1)
            weighted_sum = (out * weights).sum(dim=1)

            fused_output = self.image_space_linear(weighted_sum)

            return fused_output, attn_weights #torch.Size([160, 32, 768]), attn_weights

        elif self.mode == 'weighted_sum_only':
            if self.debug:
                logging.info("-------------------------------------------------------------------------------------------------------")
                logging.info("We are in weighted_sum_only")

            combined_seq = torch.cat([image_emb, audio_emb], dim=1)

            B, S, E = combined_seq.shape
            out = combined_seq.view(B, 2, S // 2, E)

            weights = self.additional_weights(out)
            weights = F.softmax(weights, dim=1)
            weighted_sum = (out * weights).sum(dim=1)

            fused_output = self.out_proj(weighted_sum)

            return fused_output, None #torch.Size([160, 32, 768]), attn_weights

        elif self.mode == 'cat_fusion':
            if self.debug:
                logging.info("-------------------------------------------------------------------------------------------------------")
                logging.info("We are in Cat Fusion")

            if audio_emb.shape[1] != image_emb.shape[1]:
                raise ValueError(
                    f"For cat_fusion, seq_len must match. "
                    f"Got audio {audio_emb.shape[1]}, image {image_emb.shape[1]}"
                )

            combined_seq = torch.cat([image_emb, audio_emb], dim=-1)  # [B, seq_len, 2*embed_dim]

            combined_seq_proj = self.cat_proj(combined_seq)  # [B, seq_len, embed_dim]

            attn_out, attn_weights = self.mha(
                query=combined_seq_proj,
                key=combined_seq_proj,
                value=combined_seq_proj
            )

            out = self.dropout(attn_out) + combined_seq_proj #dropout optional
            out = self.layernorm(out)

            fused_output = self.out_proj(out)

            return fused_output, attn_weights

        elif self.mode == 'x-attention':
            if self.debug:
                print("-------------------------------------------------------------------------------------------------------")
                print("We are in X Fusion")


            attn_out, attn_weights = self.mha(
                query=audio_emb,  # [B, I, 512]
                key=image_emb,  # [B, A, 512]
                value=image_emb  # [B, A, 512]
            )

            out = self.dropout(attn_out) + image_emb #dropout optional
            out = self.layernorm(out)

            fused_output = self.out_proj(out)

            return fused_output, attn_weights

        elif self.mode == 'mlp_fusion':
            if self.debug:
                logging.info("-------------------------------------------------------------------------------------------------------")
                logging.info("We are in MLP Fusion")
            combined_seq = torch.cat([image_emb, audio_emb], dim=-1) #  [B, tokens, 2* 512], [B, seq_len, 2*embed_dim]
            mlp_out = self.mlp(combined_seq)
            return mlp_out, None

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Choose from ['multimodal_sequence_fusion', 'cat_fusion', 'x-attention', 'stack_fusion_token', 'mlp_fusion', 'weighted_sum_only']."
            )

