from peft import LoraConfig, LoftQConfig
import bitsandbytes as bnb
import torch

def get_peft_config(model, rank=8, task_type="CAUSAL_LM"):
    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        r=rank,
        target_modules=target_modules,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type=task_type,
        init_lora_weights="loftq",
        loftq_config=LoftQConfig(loftq_bits=4),
    )
    return peft_config


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)