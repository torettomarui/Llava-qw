import os  
import argparse  
import torch  
from torch.utils.data import DataLoader, DistributedSampler  
from tqdm import tqdm  
from functools import partial  
from modelscope import AutoTokenizer, AutoModel  
import multiprocessing as mp  
import deepspeed  
from Datasets.data import ReyesDataset, ReyesDataCollateFn


import os

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from transformers import (AdamW, get_scheduler)
import torch
from modelscope import AutoTokenizer
import multiprocessing as mp

from Datasets.data import ReyesDataset, ReyesDataCollateFn

def train_model(train_loader, 
                model,
                tokenizer,
                device,
                epochs=1, 
                lr=1e-4, 
                save_path="./reyes_model_checkpoints",
                ):
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        i = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device)
            input_ids=batch["input_ids"].to(device)
            labels=batch["labels"].to(device)
            attention_mask=batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values, input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )
            loss = outputs.loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            train_loss += loss.item()
            if i % 200 == 0:
                print(f"loss [{loss}], Learning Rate: {current_lr:.6f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")
        
        # Save model checkpoint
        model.eval()
        output_dir = os.path.join(save_path,"epoch_{}".format(epoch+1))
        os.makedirs(output_dir, exist_ok=True)
        model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def main():  
    parser = argparse.ArgumentParser(description="Train model on specified dataset")  
    parser.add_argument("--vision_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/models/InternViT-300M-448px-V2_5", type=str, help="vision_model_path path")  
    parser.add_argument("--language_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/models/Qwen2.5-7B-Instruct", type=str, help="language_model_path path")  
    parser.add_argument("--dataset_name", default="docvqa", type=str, choices=["docvqa"], help="Dataset to train on")  
    parser.add_argument("--pretrained_mlp_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/Code/Reyes/outputs/reyes_pretrain_mlp_model_checkpoints/epoch_1", type=str, help="vision_model_path path")  
    parser.add_argument("--pretrained_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/Code/Reyes/outputs/reyes_pretrain_model_checkpoints/epoch_1", type=str, help="vision_model_path path")  
    parser.add_argument("--data_path", default="/nfs/dataset-ofs-perception/tracking/marui/Dataset/LLaVA-Finetune/llava_v1_5_mix665k.json", type=str, help="data path")  
    parser.add_argument("--image_path", default="/nfs/dataset-ofs-perception/tracking/marui/Dataset/LLaVA-Finetune/", type=str, help="image path")  
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")  
    parser.add_argument("--use_evaluate", type=bool, default=False, help="Use evaluate if this flag is passed")  
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")  
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")

    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations")  
    parser.add_argument("--save_path", type=str, default="./outputs/reyes_finetune_wo_deepspeed_model_checkpoints", help="model save path")  
    parser.add_argument("--finetune", type=bool, default=True, help="whether is finetune")  
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="Path to DeepSpeed config file")  
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")  # 添加 local_rank 参数  
    args = parser.parse_args()  


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the model  
    model = AutoModel.from_pretrained(  
        args.pretrained_model_path,  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True  
    )  

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

      
    
    print("load model successful")  
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_path, trust_remote_code=True, use_fast=False)
    
    max_num = 1  
    
    train_dataset = ReyesDataset(data_path=args.data_path, image_path=args.image_path, tokenizer=tokenizer,max_num=max_num)

    # Use DistributedSampler for multi-GPU training  
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(ReyesDataCollateFn, max_num=max_num), num_workers=6, shuffle=True,pin_memory=False)

   
    if args.finetune:  
        model.module.vision_model.requires_grad_(False)  

    # Load DeepSpeed configuration  

    #deepspeed_config.scheduler.params.total_num_steps = args.epochs * len(train_loader) // deepspeed_config.train_batch_size  

    # Initialize DeepSpeed  

    train_model(train_loader, 
                model,
                tokenizer,
                device,
                epochs=args.epochs, 
                lr=args.lr, 
                save_path=args.save_path,
                )
    
if __name__ == "__main__":  
    mp.set_start_method("spawn", force=True)  
    main()  