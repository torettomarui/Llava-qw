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

from Models.configuration_intern_vit import InternVisionConfig
from Models.modeling_intern_vit import InternVisionModel
from Models.configuration_llavaqw import LlavaQwConfig
from Models.modeling_llavaqw import LlavaQwModel
from transformers import AutoConfig, Qwen2ForCausalLM
from Datasets.data import LlavaQwDataset, LlavaQwDataCollateFn



def load_llavaqw_model(
    vision_model_path,
    language_model_path,
    use_flash_attn=True
):    
    vision_config = InternVisionConfig.from_pretrained(vision_model_path)
    llm_config = AutoConfig.from_pretrained(
        language_model_path,
        trust_remote_code=True
    )
    
    config = LlavaQwConfig(
        vision_config=vision_config.to_dict(),
        llm_config=llm_config.to_dict(),
        dynamic_image_size=True,
        force_image_size=448,
        downsample_ratio=0.5,
        use_thumbnail=True,
        select_layer=-1,
        template="chatml",
        ps_version="v2"
    )
    
    vision_model = InternVisionModel.from_pretrained(
        vision_model_path,
        config=vision_config
    )
    
    language_model = Qwen2ForCausalLM.from_pretrained(
        language_model_path,
        config=llm_config
    )
    
    model = LlavaQwModel(
        config=config,
        vision_model=vision_model,
        language_model=language_model,
        use_flash_attn=use_flash_attn
    ).to(torch.bfloat16)
    
    return model, config
        
def train_model(train_loader, 
                model,
                tokenizer,
                device,
                epochs=1, 
                lr=1e-4, 
                save_path="./LlavaQw_pretrain_model_checkpoints",
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
            if i % 50 == 0:
                print(f"loss [{loss}], Learning Rate: {current_lr:.10f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")
        
        model.eval()
        output_dir = os.path.join(save_path,"epoch_{}".format(epoch+1))
        os.makedirs(output_dir, exist_ok=True)
        model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        



    
def main():
    parser = argparse.ArgumentParser(description="Train llava-qw model on specified dataset")
    parser.add_argument("--vision_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/models/InternViT-300M-448px-V2_5", type=str, help="vision_model_path path")
    parser.add_argument("--language_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/models/Qwen2.5-7B-Instruct", type=str, help="language_model_path path")
    parser.add_argument("--data_path", default="/nfs/dataset-ofs-perception/tracking/marui/Dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json", type=str, help="data path")
    parser.add_argument("--image_path", default="/nfs/dataset-ofs-perception/tracking/marui/Dataset/LLaVA-Pretrain/images", type=str, help="image path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="./outputs/LlavaQw_pretrain_model_checkpoints", help="model save path")
    parser.add_argument("--pretrain", type=bool, default=True, help="wheather is pretrain or fintune")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model, _= load_llavaqw_model(vision_model_path=args.vision_model_path, language_model_path=args.language_model_path)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    model = model.to(device)

    print("load model sccuessful")
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_path, trust_remote_code=True, use_fast=False)
    
    train_dataset = LlavaQwDataset(data_path=args.data_path, image_path=args.image_path, tokenizer=tokenizer)

    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(LlavaQwDataCollateFn), num_workers=6, shuffle=True,pin_memory=False)

    if args.pretrain:
        model.requires_grad_(False)
        for p in model.module.mlp1.parameters():
            p.requires_grad = True

        
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