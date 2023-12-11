import tqdm
from transformers import AutoTokenizer, AutoModel, logging
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import os
import warnings
from tqdm import tqdm, trange
import torch
import transformers
import torch.optim as optim
import json
import torchvision.models as models
from PIL import Image
from copy import deepcopy
import numpy as np
from util_functions import *
from vqa_models import *
from vqa_datasets import *

logging.set_verbosity_error()

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = get_arguments()
version = f"BASE_{args.lang}_fold{args.fold}"

if not os.path.exists(f"./{get_save_path()}"):
    os.makedirs(f"./{get_save_path()}")

set_all_seed()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    def __init__(self):
        self.lr = 5e-5
        self.max_token = 50
        self.batch_size = get_batch_size()
        self.n_epoch = 50
        self.drop_out = 0.2
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = Config()

tokenizer = get_tokenizer()
train_transform, valid_transform = get_transform()
train_data, valid_data, _, _, gold_ans_list, gold_num_target = get_data(args)

train_dataset = BaselineDataset(tokenizer, train_data, gold_ans_list, config.max_token, train_transform, config)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=get_num_workers(), shuffle=True, pin_memory=True)
valid_dataset = BaselineDataset(tokenizer, valid_data, gold_ans_list, config.max_token, valid_transform, config) 
valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size , num_workers=get_num_workers(), shuffle=False, pin_memory=True)

model = BaselineModel(gold_num_target)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()
    
if __name__ == '__main__':
    best_epoch = 0
    best_acc = 0

    print(f"# Train data : {len(train_data)}")
    print(f"# Valid data : {len(valid_data)}")
    print("=======================START TRAINING=======================")
    print(f"model   : BASELINE")
    print(f"lang    : {args.lang}")
    print(f"version : {version}")

    for epoch in range(1, config.n_epoch+1):
        loss_train = AverageMeter()
        acc_train = AverageMeter()
        model.train()
        iterator = tqdm(train_loader, total=len(train_loader), unit='Iter')
        for batch in iterator: 
            batch_size = batch['answer'].size(0)

            optimizer.zero_grad()
            imgs = batch['image'].to(device)  
            q_bert_ids = batch['q_ids'].to(device)
            q_bert_mask = batch['q_mask'].to(device) 

            gold_out = model(q_bert_ids, q_bert_mask, imgs)
            answers = batch['answer'].to(device)
            answers = answers.squeeze()
            
            loss = criterion(gold_out, answers) 

            loss.backward(loss)
            optimizer.step()
            
            pred = torch.argmax(gold_out, dim=1)
            count_correct = answers.eq(pred).sum().item()
            acc = count_correct/batch_size

            # Update
            loss_train.update(loss.item(), batch_size)
            acc_train.update(acc, batch_size)
            iterator.set_description(f"[{epoch:2}/{config.n_epoch:2}]  TRAIN LOSS: {loss_train.avg:.4f}  TRAIN ACC: {acc_train.avg*100:2.2f}")

        model.eval()
        loss_valid = AverageMeter()
        acc_valid = AverageMeter()
        iterator = tqdm(valid_loader, total=len(valid_loader), unit='Iter')
        for batch in iterator:
            with torch.no_grad():
                batch_size = batch['answer'].size(0)

                imgs = batch['image'].to(device)
                q_bert_ids = batch['q_ids'].to(device)
                q_bert_mask = batch['q_mask'].to(device)
                
                answers = batch['answer'].to(device)
                answers = answers.squeeze()
                
                gold_out = model(q_bert_ids, q_bert_mask, imgs)
                loss = criterion(gold_out, answers)
            
            pred = torch.argmax(gold_out, dim=1)
            count_correct = answers.eq(pred).sum().item()
            acc = count_correct/batch_size

            # Update
            loss_valid.update(loss.item(), batch_size)
            acc_valid.update(acc, batch_size)
            iterator.set_description(f"[{epoch:2}/{config.n_epoch:2}]  VALID LOSS: {loss_valid.avg:2.4f}  VALID ACC: {acc_valid.avg*100:2.2f}")

        if acc_valid.avg*100 > best_acc:
            best_acc = acc_valid.avg*100
            best_epoch = epoch
            best_acc_model = deepcopy(model)
        
        print(f"BEST VALID ACC: {best_acc:2.2f}")

    print(f"\nSave the best acc model in epoch {best_epoch} : {best_acc}%")
    torch.save(best_acc_model.state_dict(), f"./{get_save_path()}/{version}_{best_acc:.2f}.pt")