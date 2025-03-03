import os  
import argparse  
import torch  
from torch.utils.data import DataLoader, DistributedSampler  
from tqdm import tqdm  
from functools import partial  
from modelscope import AutoTokenizer, AutoModel  
import multiprocessing as mp  
import deepspeed  
from Datasets.data import LlavaQwDataset, LlavaQwDataCollateFn
import json

def train_model(train_loader, model_engine, tokenizer, optimizer,epochs=1, save_path="./LlavaQw_model_checkpoints"):  
    for epoch in range(epochs):  
        model_engine.train()  
        train_loss = 0  
        i = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1  
            model_engine.zero_grad()  # Use model_engine's zero_grad  
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(model_engine.device)
            input_ids = batch["input_ids"].to(model_engine.device)  
            labels = batch["labels"].to(model_engine.device)  
            attention_mask = batch["attention_mask"].to(model_engine.device)  

            outputs = model_engine(  
                pixel_values=pixel_values, input_ids=input_ids, labels=labels, attention_mask=attention_mask  
            )  
            loss = outputs.loss.mean()  

            model_engine.backward(loss)  
            model_engine.step()  
            current_lr = optimizer.param_groups[0]['lr']
            train_loss += loss.item()
            if i % 50 == 0:  
                print(f"loss [{loss}], Learning Rate: {current_lr:.10f}")
            

        avg_train_loss = train_loss / len(train_loader)  
        print(f"Average Training Loss: {avg_train_loss}")  
        
        # Save model checkpoint  
        model_engine.eval()  
        output_dir = os.path.join(save_path, "epoch_{}".format(epoch + 1))  
        os.makedirs(output_dir, exist_ok=True)  
        model_engine.save_checkpoint(output_dir)  
        tokenizer.save_pretrained(output_dir)   

def main():  
    parser = argparse.ArgumentParser(description="Train model on specified dataset")  
    parser.add_argument("--language_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/models/Qwen2.5-7B-Instruct", type=str, help="language_model_path path")  
    parser.add_argument("--pretrained_model_path", default="/nfs/dataset-ofs-perception/tracking/marui/Code/llava-qw/outputs/LlavaQw_pretrain_model_checkpoints/epoch_1", type=str, help="vision_model_path path")  
    parser.add_argument("--data_path", default="/nfs/dataset-ofs-perception/tracking/marui/Dataset/LLaVA-Finetune/llava_v1_5_mix665k.json", type=str, help="data path")  
    parser.add_argument("--image_path", default="/nfs/dataset-ofs-perception/tracking/marui/Dataset/LLaVA-Finetune/", type=str, help="image path")  
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")  
    parser.add_argument("--save_path", type=str, default="./outputs/LlavaQw_finetune_model_checkpoints", help="model save path")  
    parser.add_argument("--finetune", type=bool, default=True, help="whether is finetune")  
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="Path to DeepSpeed config file")  
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")  # 添加 local_rank 参数  
    args = parser.parse_args()  


    with open(args.deepspeed_config, 'r') as file:  
        deepspeed_json = json.load(file)

    # Initialize DeepSpeed  
    deepspeed.init_distributed()  

    # Set device based on local_rank  
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")  
    torch.cuda.set_device(device)

    # Load the model  
    model = AutoModel.from_pretrained(  
        args.pretrained_model_path,  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True  
    )  

    model = model.to(device)
    
    print("load model successful")  
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_path, trust_remote_code=True, use_fast=False)  
    
    train_dataset = LlavaQwDataset(data_path=args.data_path, image_path=args.image_path, tokenizer=tokenizer)  

    # Use DistributedSampler for multi-GPU training  
    train_sampler = DistributedSampler(train_dataset)  
    train_loader = DataLoader(train_dataset, batch_size=deepspeed_json["train_micro_batch_size_per_gpu"], collate_fn=partial(LlavaQwDataCollateFn), num_workers=12, shuffle=False, sampler=train_sampler, pin_memory=True)  

    if args.finetune:  
        model.vision_model.requires_grad_(False)  


    deepspeed_json["scheduler"]["params"]["total_num_steps"] = args.epochs * len(train_loader) // deepspeed_json["train_batch_size"]

    # Initialize DeepSpeed  
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=deepspeed_json)  

    train_model(train_loader, model_engine, tokenizer, optimizer, epochs=args.epochs, save_path=args.save_path)  
    
if __name__ == "__main__":  
    mp.set_start_method("spawn", force=True)  
    main()  