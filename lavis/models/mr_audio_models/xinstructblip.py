import logging
import os
import torch.nn as nn
import torch
import random
import contextlib

from torch.nn.modules.module import _IncompatibleKeys
from lavis.common.registry import registry
from transformers import LlamaTokenizer
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from lavis.common.utils import is_url
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.common.dist_utils import download_cached_file
from lavis.models.eva_vit import create_eva_vit_g
from transformers import BertTokenizer
from lavis.models.blip2_mr_models.utils import post_process

from peft import get_peft_model

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_input_ones = input_atts[i].sum()
        input_part_targets_len.append(this_input_ones)
        llm_tokens['input_ids'].append(
            torch.cat([
                input_ids[i][:this_input_ones],
                output_ids[i][1:],
                input_ids[i][this_input_ones:]
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                input_atts[i][:this_input_ones],
                output_atts[i][1:],
                input_atts[i][this_input_ones:]
            ])
        )
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len

@registry.register_model("blip2_vicuna_xinstruct_mr")
class XInstructBLIP(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/empty.yaml",
    }

    def __init__(
        self,
        model_path,
        audio_path,
        enumerate_inputs=False,
        modalities=["audio", "video"],
        lora=True,
        interleave_seconds=True,
        finetuned="",  
    ):
        super().__init__()

        self.enumerate_inputs = enumerate_inputs
        self.modalities = modalities 
        self.lora = lora
        self.interleave_seconds = interleave_seconds
        self.finetuned = finetuned # Path to finetuned checkpoint
        self.max_txt_len=128
        self.max_output_txt_len=64
        self.max_new_tokens=64

        # Init video encoder
        self.pretrained_video_qformer = "https://storage.googleapis.com/sfr-xinstructblip-data-research/model/xinstructblip_checkpoints/vicuna7b/video_qformer.pth"
        video_encoder_kwargs = {
            "image_size": 224,
            "drop_path_rate": 0,
            "use_grad_checkpoint": False,
            "load_ln_path": self.pretrained_video_qformer,
            "load_ln_type": "video"
        }
        self.video_encoder, self.video_ln = self.init_video_encoder("eva_clip_g", precision="fp16", **video_encoder_kwargs)
        
        # Freeze video encoder
        for name, param in self.video_encoder.named_parameters():
            param.requires_grad = False
        self.video_encoder = self.video_encoder.eval()
        self.video_encoder.train = disabled_train


        if "audio" in self.modalities: # For baselines w/o audio
            # Init audio encoder
            self.pretrained_audio_qformer = "https://storage.googleapis.com/sfr-xinstructblip-data-research/model/xinstructblip_checkpoints/vicuna7b/audio_qformer_improved.pth"

            checkpoint_path = audio_path
            load_ln_path = self.pretrained_video_qformer
            load_ln_type = "audio"

            self.audio_encoder, self.audio_ln = self.init_audio_encoder("beats",
                                                                        precision="fp16",
                                                                        checkpoint_path=checkpoint_path,
                                                                        load_ln_path=load_ln_path,
                                                                        load_ln_type=load_ln_type)

            # Freeze audio encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder = self.audio_encoder.eval()
            self.audio_encoder.train = disabled_train



        ##### Init QFormers ####
        self.tokenizer = self.init_tokenizer(truncation_side="left") # 30523 tokens. 
        self.num_query_token = 32
        
        for modality in self.modalities:
            modality_num_features = getattr(self, f"{modality}_encoder").num_features
            
            logging.info(f"Initializing {modality} QFormer and query tokens of length {self.num_query_token}")
            modality_qformer, modality_query_tokens = self.init_Qformer(
                self.num_query_token, 
                modality_num_features,
                pretrained_qformer=getattr(self, f"pretrained_{modality}_qformer"),
                load_attention=True,
                load_qformer_type=modality
            ) 

            modality_qformer.resize_token_embeddings(len(self.tokenizer))
            modality_qformer.cls = None
            setattr(self, f"{modality}_Qformer", modality_qformer)
            setattr(self, f"{modality}_query_tokens", modality_query_tokens)

        
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})


        if self.lora:
            from lavis.models.mr_audio_models.utils import get_peft_config
            self.llm_model = LlamaForCausalLM.from_pretrained(
                model_path, 
                load_in_8bit=True,
                torch_dtype=torch.float16
            )
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            
            # reduce memory usage
            self.llm_model.gradient_checkpointing_enable() # does not work with find_unused_parameters = True
            self.llm_model.enable_input_require_grads()
            self.llm_model.lm_head = CastOutputToFloat(self.llm_model.lm_head)
            self.llm_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
            self.llm_hidden_size = self.llm_model.config.hidden_size

            self.llm_model = get_peft_model(model=self.llm_model,peft_config=get_peft_config(self.llm_model))

            # LLM frozen by peft by default
        
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            )
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            self.llm_hidden_size = self.llm_model.config.hidden_size
            # Freeze LLM
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False


        for modality in self.modalities:
            load_projection_path = getattr(self, f"pretrained_{modality}_qformer")
            setattr(self, f"load_projection_{modality}", load_projection_path)

            qformer = getattr(self, f"{modality}_Qformer")
            proj = self.init_vicuna_projection(
                qformer.config.hidden_size, 
                self.llm_hidden_size,
                load_projection_path=load_projection_path, 
                load_projection_type=modality
            )
            setattr(self, f"{modality}_llm_proj", proj)
        
        if not self.finetuned:
            self.load_from_pretrained("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth")
        else:
            self.load_checkpoint(self.finetuned)
            
        # Freeze QFormers
        for modality in self.modalities:
            for name, param in getattr(self, f"{modality}_ln").named_parameters():
                param.requires_grad = False
            getattr(self, f"{modality}_query_tokens").requires_grad = False
            for name, param in getattr(self, f'{modality}_Qformer').named_parameters():
                param.requires_grad = False
            for name, param in getattr(self, f'{modality}_llm_proj').named_parameters():
                param.requires_grad = False

        self.MODALITY_TO_CUE = {
            "video": " video: ",
            "audio": " audio: ",
        }
        
        self.tokenized_cue = {}
        self.emb_cue = {}
        self.att_cue = {}
        for modality in self.modalities:
            self.tokenized_cue[modality] = self.llm_tokenizer(self.MODALITY_TO_CUE[modality], return_tensors="pt")
            self.emb_cue[modality] = self.llm_model.get_input_embeddings()(self.tokenized_cue[modality].input_ids.to(self.device))
            self.att_cue[modality] = self.tokenized_cue[modality].attention_mask.to(self.device)

        

    @torch.no_grad()
    def generate(self, samples):
        self.llm_tokenizer.padding_side = "left"
        curr_modalities = self.modalities
        bs = samples["video"].size(0)
        prompt = samples["text_input"]

        query_tokens = {}
        for modality in curr_modalities:
            query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(bs, -1, -1)
        

        text_Qformer = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)



        Qformer_atts = {}
        query_atts = {}
        
        for modality in curr_modalities:
            # B, Token Size
            query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
            # B, Token Size + Inp Size
            Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)


        embeds = {}
        data_atts = {}
        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video":
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))

            elif modality == 'audio':
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))


        query_outputs = {}
        num = {}
        for modality in curr_modalities:
            num[modality] = len(embeds[modality])
            bs = embeds[modality][0].shape[0]
            indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
            reordered_embeds = torch.cat(embeds[modality])[indices]
            reordered_atts = torch.cat(data_atts[modality])[indices]
            query_output = getattr(self, f"{modality}_Qformer").bert(
                text_Qformer.input_ids.repeat(num[modality], 1),
                attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                encoder_hidden_states=reordered_embeds,
                encoder_attention_mask=reordered_atts,
                return_dict=True,
            )
            query_outputs[modality] = query_output
           


        inputs_llm = {}
        atts_llm = {}

        for i,modality in enumerate(curr_modalities):
            # num*bs, num query tokens, llm emb size
            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
            # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
            inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs,  num[modality]*self.num_query_token, -1)
            atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            

        ## remove trailing whitespace 
        prompt = [p.strip() for p in prompt]

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        bs = llm_tokens.input_ids.shape[0]

        # Get timestamp embeds
        if self.interleave_seconds:
            flattened_timestamps = [
                f" {t} "
                for timestamps in samples["timestamps"]
                for t in timestamps
            ]
            timestamp_tokens = self.llm_tokenizer(
                flattened_timestamps,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            timestamp_inputs_llm = self.llm_model.get_input_embeddings()(timestamp_tokens.input_ids)
            timestamp_atts_llm = timestamp_tokens.attention_mask.to(self.device)

            timestamp_inputs_llm = timestamp_inputs_llm.view(len(samples["timestamps"]), len(samples["timestamps"][0]), *timestamp_inputs_llm.shape[1:]) # batch_size X timestamps_per_batch X tokens_per_timestamp X embed_dim (= 4096)
            timestamp_atts_llm = timestamp_atts_llm.view(len(samples["timestamps"]), len(samples["timestamps"][0]), timestamp_atts_llm.shape[-1]) # batch_size X timestamps_per_batch X tokens_per_timestamp



        att_list = []
        inp_list = []        

        # Joint video audio
        for pos in range(num['video']):
            if self.enumerate_inputs:
                enumeration_pos = self.llm_tokenizer(
                    [f"{'' if pos == 0 else ' '}({chr(97+pos)}) " for _ in prompt],
                    return_tensors="pt",
                    add_special_tokens=False if (pos!= 0) else True
                ).to(self.device)
                enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration_pos.input_ids)
                enumeration_atts_llm = enumeration_pos.attention_mask.to(self.device)
                inp_list.extend([enumeration_inputs_llm])
                att_list.extend([enumeration_atts_llm])
            
            # Cues
            for modality in ['video', 'audio']:
                att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])

            if self.interleave_seconds:
                inp_list.extend([timestamp_inputs_llm[:, pos, :, :]])
                att_list.extend([timestamp_atts_llm[:, pos, :]])

        # Duration
        duration_tokens = self.llm_tokenizer(
            [f"{dur} " for dur in samples["duration"]],
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        att_list.append(duration_tokens.attention_mask)
        duration_inputs_llm = self.llm_model.get_input_embeddings()(duration_tokens.input_ids)
        inp_list.append(duration_inputs_llm)
                

        att_list.append(llm_tokens.attention_mask)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inp_list.append(inputs_embeds)
       
        attention_mask = torch.cat(att_list, dim=1)
        inputs_embeds = torch.cat(inp_list, dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
            )
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [o.strip() for o in output_text]
        print(output_text)

        return {
            "qid": samples["query_id"],
            "prediction": [post_process(out) for out in output_text],
            "raw_prediction": output_text,
            "answer": samples["relevant_windows"],
            "duration": samples["duration"].tolist() if isinstance(samples["duration"], torch.Tensor) else samples["duration"],
        }
    
    def forward(self, samples):
        
        if samples is None or samples == {} or not any([modality in samples for modality in self.modalities]):
            return {"loss": torch.tensor(0.0)}

        random.shuffle(self.modalities)

        curr_modalities = [modality for modality in self.modalities if modality in samples]
        
        
        embeds = {}
        query_tokens = {}
        data_atts = {}
        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video":
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)

            elif modality == 'audio':
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)

        query_outputs = {}
        text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
        
        
        Qformer_atts = {}
        query_atts = {}
        num = {}
        for modality in curr_modalities:
            # B, Token Size
            query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
            # B, Token Size + Inp Size
            Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)
            num[modality] = len(embeds[modality])
            bs = embeds[modality][0].shape[0]
            indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
            reordered_embeds = torch.cat(embeds[modality])[indices]
            reordered_atts = torch.cat(data_atts[modality])[indices]
            query_output = getattr(self, f"{modality}_Qformer").bert(
                text_Qformer.input_ids.repeat(num[modality], 1),
                attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                encoder_hidden_states=reordered_embeds,
                encoder_attention_mask=reordered_atts,
                return_dict=True,
            )
            query_outputs[modality] = query_output
                    
        
        inputs_llm = {}
        atts_llm = {}
        for modality in curr_modalities:
            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
            # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
            inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
            atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
    

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=True
        ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['relevant_windows']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len = concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        bs = inputs_embeds.shape[0]
        
        prompt = samples["text_input"]

        # Get timestamp embeds
        if self.interleave_seconds:
            flattened_timestamps = [
                f" {t} "
                for timestamps in samples["timestamps"]
                for t in timestamps
            ]
            timestamp_tokens = self.llm_tokenizer(
                flattened_timestamps,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            timestamp_inputs_llm = self.llm_model.get_input_embeddings()(timestamp_tokens.input_ids)
            timestamp_atts_llm = timestamp_tokens.attention_mask.to(self.device)

            timestamp_inputs_llm = timestamp_inputs_llm.view(len(samples["timestamps"]), len(samples["timestamps"][0]), *timestamp_inputs_llm.shape[1:]) # batch_size X timestamps_per_batch X tokens_per_timestamp X embed_dim (= 4096)
            timestamp_atts_llm = timestamp_atts_llm.view(len(samples["timestamps"]), len(samples["timestamps"][0]), timestamp_atts_llm.shape[-1]) # batch_size X timestamps_per_batch X tokens_per_timestamp

        att_list = []
        inp_list = []
        # joint video audio TODO add seconds
        for pos in range(num['video']):
            if self.enumerate_inputs:
                enumeration_pos = self.llm_tokenizer(
                    [f"{'' if pos == 0 else ' '}({chr(97+pos)}) " for _ in prompt],
                    return_tensors="pt",
                    add_special_tokens=False if (pos!= 0) else True
                ).to(self.device)
                enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration_pos.input_ids)
                enumeration_atts_llm = enumeration_pos.attention_mask.to(self.device)
                inp_list.extend([enumeration_inputs_llm])
                att_list.extend([enumeration_atts_llm])

            for modality in ['video', 'audio']:
                att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
             
            if self.interleave_seconds:
                inp_list.extend([timestamp_inputs_llm[:, pos, :, :]])
                att_list.extend([timestamp_atts_llm[:, pos, :]])


        # Duration
        duration_tokens = self.llm_tokenizer(
            [f"{dur} " for dur in samples["duration"]],
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        att_list.append(duration_tokens.attention_mask)
        duration_inputs_llm = self.llm_model.get_input_embeddings()(duration_tokens.input_ids)
        inp_list.append(duration_inputs_llm)
             
        
        
        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(torch.cat(att_list, dim=1).size(), dtype=torch.long).to(self.device).fill_(-100)
        )

        # append llm prompt + output to queries
        att_list.append(llm_tokens['attention_mask'])
        inp_list.append(inputs_embeds)
        
        inputs_embeds = torch.cat(inp_list, dim=1)
        attention_mask = torch.cat(att_list, dim=1)
        targets = torch.cat([empty_targets, targets], dim=1)

       
        
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return {"loss": outputs.loss}
    
    @classmethod
    def from_config(cls, cfg):
        model_path = cfg.get("model_path")
        audio_path = cfg.get("audio_path")
        enumerate_inputs = cfg.get("enumerate_inputs", False)
        modalities = cfg.get("modalities", ["audio", "video"])
        lora = cfg.get("lora", True)
        interleave_seconds = cfg.get("interleave_seconds", True)
        finetuned = cfg.get("finetuned", "")  

        model = cls(
            model_path=model_path,
            audio_path=audio_path,
            enumerate_inputs=enumerate_inputs,
            modalities=modalities,
            lora=lora,
            interleave_seconds=interleave_seconds,
            finetuned=finetuned,
        )

        return model


    @classmethod
    def init_Qformer(cls, num_query_token, modality_width, cross_attention_freq=2, pretrained_qformer=None, load_attention=False, load_qformer_type=""):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = modality_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.vocab_size += 1 # for special token [DEC]
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        if pretrained_qformer:
            url_or_filename=pretrained_qformer
            logging.info(f"Loading pretrained qformer weights and query tokens from {url_or_filename} of type {load_qformer_type}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            
            if load_qformer_type:
                load_qformer_type = f"{load_qformer_type}_"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if load_qformer_type+'Qformer.' in k:
                    if not load_attention and 'attention' in k:
                        continue
                    loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            Qformer.load_state_dict(loaded_state_dict, strict=False)
            query_tokens.data = checkpoint[load_qformer_type+'query_tokens']
        
        return Qformer, query_tokens


    def init_video_encoder(self, model_name, precision, **kwargs):
        assert model_name == "eva_clip_g"
        video_encoder = create_eva_vit_g(
                kwargs['image_size'], kwargs['drop_path_rate'], kwargs['use_grad_checkpoint'], precision
            )

        ln_video = self.init_ln(video_encoder.num_features, load_ln_path=kwargs['load_ln_path'], load_ln_type=kwargs['load_ln_type'])

        return video_encoder, ln_video



    def init_audio_encoder(self, model_name, precision,checkpoint_path,load_ln_path,load_ln_type):
        assert model_name == "beats"
        from lavis.models.beats_encoder import BeatsEncoder
        audio_encoder = BeatsEncoder(checkpoint_path=checkpoint_path)
        ln_audio = self.init_ln(audio_encoder.num_features, load_ln_path=load_ln_path, load_ln_type=load_ln_type)

        return audio_encoder, ln_audio

    @classmethod
    def init_ln(cls, num_features, load_ln_path:str=False, load_ln_type=""):
        ln = LayerNorm(num_features)
        if load_ln_path and load_ln_type:
            url_or_filename=load_ln_path
            logging.info(f"Loading pretrained layer norm weights from {url_or_filename} of type {load_ln_type}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            
            if load_ln_type:
                load_ln_type = f"{load_ln_type}_ln" if "vision" not in load_ln_type else "ln_vision"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if load_ln_type in k:
                    loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            ln.load_state_dict(loaded_state_dict, strict=False)
        
        return ln
    
    @classmethod
    def init_vicuna_projection(cls, input_size, output_size, load_projection_path=False, load_projection_type="", projection_key=None):
        proj = nn.Linear(input_size, output_size)
        if load_projection_path:
            url_or_filename=load_projection_path
            logging.info(f"Loading pretrained projection weights from {url_or_filename} of type {load_projection_type} with key {projection_key if projection_key else load_projection_type+'_llm_proj.'}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            if load_projection_type:
                load_projection_type = f"{load_projection_type}_"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if projection_key:
                    if projection_key in k:
                        loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
                else:
                    if load_projection_type+'llm_proj.' in k:
                        loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            proj.load_state_dict(loaded_state_dict, strict=False)
        
        return proj
    
    def get_state_dict(self, url_or_filename, **kwargs):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        return state_dict
    
    def load_from_pretrained(self, url_or_filename, **kwargs):
        state_dict = self.get_state_dict(url_or_filename)
        self.load_state_dict(state_dict, strict=False)
        logging.info("load checkpoint from %s" % url_or_filename)

    def load_checkpoint(self, url_or_filename, **kwargs):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """
        state_dict = self.get_state_dict(url_or_filename)
        self.load_state_dict(state_dict, strict=True)
        logging.info("load checkpoint from %s" % url_or_filename)
    
    def load_state_dict(self, state_dict, strict=True):
        # from pdb import set_trace; set_trace()
        unexpected_keys = []
        missing_keys = []
        
        for modality in self.modalities:
            ## Load Q-Former if not loaded from config
            if not getattr(self, f"pretrained_{modality}_qformer"):

                modality_qformer_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_Qformer" == k.split('.')[0]}
                msg = getattr(self, f"{modality}_Qformer").load_state_dict(modality_qformer_state_dict, strict=strict)
                missing_keys.extend(msg.missing_keys)
                unexpected_keys.extend(msg.unexpected_keys)

                ## Load query tokens
                if f"{modality}_query_tokens" not in state_dict:
                    missing_keys.append(f"{modality}_query_tokens")
                else:
                    logging.info(f"Loaded {modality} query tokens")
                    getattr(self, f"{modality}_query_tokens").data =  state_dict[f"{modality}_query_tokens"]

                # load modality layer norm if not loaded from config
                modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_ln" in k.split('.')[0]}

                msg = getattr(self, f"{modality}_ln").load_state_dict(modality_ln_dict, strict=strict)
                missing_keys.extend(msg.missing_keys)
                unexpected_keys.extend(msg.unexpected_keys)

            ## Load LLM projections if not loaded from config
            if not getattr(self, f"load_projection_{modality}"):
                if not getattr(self, f"projection_path_{modality}"):
                    logging.info(f"Loaded {modality} llm  projection")
                    modality_llm_projection_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_llm_proj" in k.split('.')[0]}
                    msg = getattr(self, f"{modality}_llm_proj").load_state_dict(modality_llm_projection_dict, strict=strict)
                    missing_keys.extend(msg.missing_keys)
                    unexpected_keys.extend(msg.unexpected_keys)
        
        ## llm model is loaded from pretrained
        lora_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"llm_model" in k.split('.')[0]}

        if not self.lora or len(lora_state_dict) == 0:
            unexpected_keys = [k for k in unexpected_keys if k.split('.')[0] != 'llm_model']
        else:
            msg = self.llm_model.load_state_dict({'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"llm_model" in k.split('.')[0]}, strict=False)
            missing_keys.extend(["llm_model."+k for k in msg.missing_keys])
        missing_keys = [k for k in missing_keys if 'encoder' not in k.split('.')[0]]
        missing_keys = [k for k in missing_keys if k.split('.')[0] != 'llm_model']
        return _IncompatibleKeys(missing_keys, unexpected_keys)
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        from lavis.models.base_model import BaseModel
        return BaseModel.get_optimizer_params(self, weight_decay, lr_scale=lr_scale)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


