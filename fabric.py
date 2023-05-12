"""
lightning run model fabric.py --strategy=deepspeed_stage_3 --devices=4 --accelerator=cuda
"""

import os, time, math, pickle, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import lightning as L
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from lightning.fabric.strategies import DeepSpeedStrategy
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from mpt_7b_instruct.model.blocks import MPTBlock

class LLM_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        inputs = row['input']
        targets = row['target']
        
        input_encoding = self.tokenizer.encode_plus(inputs,
                                                    max_length=self.max_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt',)

        input_ids = input_encoding['input_ids']

        target_encoding = self.tokenizer.encode_plus(targets,
                                                    max_length=self.max_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt')
        
        labels = torch.where(target_encoding['input_ids'] == 0, torch.tensor(-100), target_encoding['input_ids'])

        return {'input_ids': input_ids, 'labels': labels}
# -----------------------------------------------------------------------------
def main():
    L.seed_everything(42)

    fabric = L.Fabric(num_nodes=1, devices=4, precision="bf16-mixed", strategy='deepspeed_stage_3')

    max_length = 256
    model_name = 'mosaicml/mpt-7b-instruct'
    tokenizer_name =  "EleutherAI/gpt-neox-20b"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, 
                                                model_max_length=max_length, 
                                                use_fast=True)

    # Data
    tokenizer.pad_token = tokenizer.eos_token

    with open('./CodeAlpaca-20k.json', 'rb') as f:
            dataset = json.load(f)

    train_data, val_data = train_test_split(dataset, test_size=0.2)
    train_dataset = LLM_Dataset(train_data, tokenizer)
    val_dataset = LLM_Dataset(val_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

    # Model
    config = AutoConfig.from_pretrained(model_name,trust_remote_code=True)
    config.attn_config['attn_impl'] = 'torch'

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config, 
                                                torch_dtype=torch.bfloat16,
                                                # low_cpu_mem_usage=True,
                                                trust_remote_code=True)

    # Optimizer
    optimizer = FusedAdam(model.parameters(), lr=1e-3)

    model, optimizer = fabric.setup(model, optimizer)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    train(fabric, model, optimizer, train_dataloader, val_dataloader)

def train(fabric, model, optimizer, train_dataloader, val_dataloader, max_epochs=20):
    for epoch in range(max_epochs):
        train_epoch(fabric, model, optimizer, train_dataloader, epoch)

def train_epoch(fabric, model, optimizer, train_dataloader, epoch):
    for batch_idx, batch in enumerate(train_dataloader):

        input, target = batch['input_ids'], batch['labels']

        loss = model(input_ids=input, labels=target).loss

        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, clip_val=0.25)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 1 == 0:
            fabric.print(f"epoch: {epoch} - iteration: {batch_idx} - loss {loss.item():.4f}")

if __name__ == "__main__":
    main()