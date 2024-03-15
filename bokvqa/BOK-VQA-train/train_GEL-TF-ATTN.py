import tqdm
import torch
from transformers import AutoTokenizer, logging
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torchkge.models import ConvKBModel
import pickle 
import transformers
import os
import warnings
import torchvision.models as models  
from tqdm import tqdm, trange
import re
import random
import numpy as np
import json
from copy import deepcopy
from util_functions import *
from vqa_models import *
from vqa_datasets import *

logging.set_verbosity_error()

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = get_arguments()
version = f"GEL-TF-ATTN_{args.lang}_fold{args.fold}"

if not os.path.exists(f"./{get_save_path()}"):
    os.makedirs(f"./{get_save_path()}")

set_all_seed()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    def __init__(self):
        self.kge_n_iter = 50000
        self.kge_lr = 1e-4
        self.kge_batch = 512
        self.kge_margin = 0.5
        self.kge_conv_size = 3

        self.lr = 5e-5
        self.max_token = 50
        self.batch_size = get_batch_size()
        self.n_epoch = 50
        self.drop_out = 0.2
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = Config()

tokenizer = get_tokenizer()
train_transform, valid_transform = get_transform()
train_data, valid_data, triple_ans_list, triple_num_target, gold_ans_list, gold_num_target = get_data(args)
KGEModel, kg, emb_entity_, emb_rel_ = get_KGE(config, args.kge_data, args.kge_model)

train_dataset = GELVQADataset(tokenizer, train_data, gold_ans_list, triple_ans_list, config.max_token, train_transform, 'train', config, emb_entity_, emb_rel_, kg)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=get_num_workers(), shuffle=True, pin_memory=True)
valid_dataset = GELVQADataset(tokenizer, valid_data, gold_ans_list, triple_ans_list, config.max_token, valid_transform, 'train', config, emb_entity_, emb_rel_, kg)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size , num_workers=get_num_workers(), shuffle=False, pin_memory=True)

model = GELVQAAttn(gold_num_target, triple_num_target)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    best_epoch = 0
    best_acc = 0
    print(f"# Train data : {len(train_data)}")
    print(f"# Valid data : {len(valid_data)}")
    print("=======================START TRAINING=======================")
    print(f"model   : GEL-VQA")
    print(f"lang    : {args.lang}")
    print(f"version : {version}")

    for epoch in range(1, config.n_epoch+1):
        loss_train = AverageMeter()
        g_loss_train = AverageMeter()
        h_loss_train = AverageMeter()
        r_loss_train = AverageMeter()
        t_loss_train = AverageMeter()

        g_acc_train = AverageMeter()
        h_acc_train = AverageMeter()
        r_acc_train = AverageMeter()
        t_acc_train = AverageMeter()
        hrt_acc_train = AverageMeter()

        model.train()
        iterator = tqdm(enumerate(train_loader), total=len(train_loader), unit='Iter')
        for idx, batch in iterator:
            batch_size = batch['answer'].size(0)

            optimizer.zero_grad()
            imgs = batch['image'].to(device)
            q_bert_ids = batch['q_ids'].to(device)
            q_bert_mask = batch['q_mask'].to(device)

            g_label = batch['answer'].to(device).squeeze()
            h_label = batch['h_label'].to(device).squeeze()
            r_label = batch['r_label'].to(device).squeeze()
            t_label = batch['t_label'].to(device).squeeze()

            # Teacher Forcing
            head = [triple_ans_list['h'][i] for i in h_label]
            rel = [triple_ans_list['r'][i] for i in r_label]
            tail = [triple_ans_list['t'][i] for i in t_label]

            emb = get_embedded_vec(batch_size, head, rel, tail, emb_entity_, emb_rel_, kg)
            emb = emb.to(config.device)

            gold_out, h_out, r_out, t_out, _, emb_attn_output_weight = model(q_bert_ids, q_bert_mask, imgs, emb, get_gold=True, bs=batch_size) 

            loss_g = criterion(gold_out, g_label) 
            loss_h = criterion(h_out, h_label)
            loss_r = criterion(r_out, r_label)
            loss_t = criterion(t_out, t_label)

            loss = loss_g + loss_h + loss_r + loss_t
            loss.backward(loss)
            optimizer.step()
            loss_train.update(loss.item(), batch_size)
            g_loss_train.update(loss_g.item(), batch_size)
            h_loss_train.update(loss_h.item(), batch_size)
            r_loss_train.update(loss_r.item(), batch_size)
            t_loss_train.update(loss_t.item(), batch_size)

            g_pred = torch.argmax(gold_out, dim=1)
            g_count_correct = g_label.eq(g_pred).sum().item()
            g_acc = g_count_correct/batch_size
            
            h_pred = torch.argmax(h_out, dim=1)
            h_count_correct = h_label.eq(h_pred).sum().item() 
            h_acc = h_count_correct/batch_size

            r_pred = torch.argmax(r_out, dim=1)
            r_count_correct = r_label.eq(r_pred).sum().item()
            r_acc = r_count_correct/batch_size

            t_pred = torch.argmax(t_out, dim=1)
            t_count_correct = t_label.eq(t_pred).sum().item()
            t_acc = t_count_correct/batch_size

            hrt_count_correct = (torch.stack((h_label, r_label, t_label)).eq(torch.stack((h_pred, r_pred, t_pred))).float().sum(dim=0) == 3).sum().item()
            hrt_acc = hrt_count_correct/batch_size
            
            # Update
            g_acc_train.update(g_acc, batch_size)
            h_acc_train.update(h_acc, batch_size)
            r_acc_train.update(r_acc, batch_size)
            t_acc_train.update(t_acc, batch_size)
            hrt_acc_train.update(hrt_acc, batch_size)

            iterator.set_description(f"[{epoch:2}/{config.n_epoch:2}]  [TRAIN LOSS: {loss_train.avg:.4f}]  [TRAIN ACC: {g_acc_train.avg*100:2.2f}]  [Head : {h_acc_train.avg*100:2.2f}]  [Rel : {r_acc_train.avg*100:2.2f}]  [Tail : {t_acc_train.avg*100:2.2f}]  [ALL : {hrt_acc_train.avg*100:2.2f}]")
        

        loss_valid = AverageMeter()
        g_loss_valid = AverageMeter()
        h_loss_valid = AverageMeter()
        r_loss_valid = AverageMeter()
        t_loss_valid = AverageMeter()

        g_acc_valid = AverageMeter()
        h_acc_valid = AverageMeter()
        r_acc_valid = AverageMeter()
        t_acc_valid = AverageMeter()
        hrt_acc_valid = AverageMeter()
        model.eval()

        iterator = tqdm(valid_loader, total=len(valid_loader), unit='Iter')
        for batch in iterator:
            with torch.no_grad():
                batch_size = batch['answer'].size(0)
                imgs = batch['image'].to(device)
                q_bert_ids = batch['q_ids'].to(device)
                q_bert_mask = batch['q_mask'].to(device)

                g_label = batch['answer'].to(device).squeeze()
                h_label = batch['h_label'].to(device).squeeze()
                r_label = batch['r_label'].to(device).squeeze()
                t_label = batch['t_label'].to(device).squeeze()

                '''
                predict triples
                '''
                h_out, r_out, t_out, _ = model(q_bert_ids, q_bert_mask, imgs, emb=0, get_gold=False, bs=batch_size) 
                h_pred = torch.argmax(h_out, dim=1)
                r_pred = torch.argmax(r_out, dim=1)
                t_pred = torch.argmax(t_out, dim=1)

                head = [triple_ans_list['h'][i] for i in h_pred]
                rel = [triple_ans_list['r'][i] for i in r_pred]
                tail = [triple_ans_list['t'][i] for i in t_pred]
                emb = get_embedded_vec(batch_size, head, rel, tail, emb_entity_, emb_rel_, kg)
                emb = emb.to(device)

                gold_out, h_out, r_out, t_out, _, attn_output_weight = model(q_bert_ids, q_bert_mask, imgs, emb, get_gold=True, bs=batch_size)
                
                loss_g = criterion(gold_out, g_label) 
                loss_h = criterion(h_out, h_label)
                loss_r = criterion(r_out, r_label)
                loss_t = criterion(t_out, t_label)

                loss = loss_g + loss_h + loss_r + loss_t
            loss_valid.update(loss.item(), batch_size)
            g_loss_valid.update(loss_g.item(), batch_size)
            h_loss_valid.update(loss_h.item(), batch_size)
            r_loss_valid.update(loss_r.item(), batch_size)
            t_loss_valid.update(loss_t.item(), batch_size)

            g_pred = torch.argmax(gold_out, dim=1)
            g_count_correct = g_label.eq(g_pred).sum().item()
            g_acc = g_count_correct/batch_size
            
            h_pred = torch.argmax(h_out, dim=1)
            h_count_correct = h_label.eq(h_pred).sum().item() 
            h_acc = h_count_correct/batch_size

            r_pred = torch.argmax(r_out, dim=1)
            r_count_correct = r_label.eq(r_pred).sum().item()
            r_acc = r_count_correct/batch_size

            t_pred = torch.argmax(t_out, dim=1)
            t_count_correct = t_label.eq(t_pred).sum().item()
            t_acc = t_count_correct/batch_size

            hrt_count_correct = (torch.stack((h_label, r_label, t_label)).eq(torch.stack((h_pred, r_pred, t_pred))).float().sum(dim=0) == 3).sum().item()
            hrt_acc = hrt_count_correct/batch_size

            g_acc_valid.update(g_acc, batch_size)
            h_acc_valid.update(h_acc, batch_size)
            r_acc_valid.update(r_acc, batch_size)
            t_acc_valid.update(t_acc, batch_size)
            hrt_acc_valid.update(hrt_acc, batch_size)

            iterator.set_description(f"[{epoch:2}/{config.n_epoch:2}]  [VALID LOSS: {loss_valid.avg:2.4f}]  [VALID ACC: {g_acc_valid.avg*100:2.2f}]  [Head : {h_acc_valid.avg*100:2.2f}]  [Rel : {r_acc_valid.avg*100:2.2f}]  [Tail : {t_acc_valid.avg*100:2.2f}]  [ALL : {hrt_acc_valid.avg*100:2.2f}]")

        if g_acc_valid.avg*100 > best_acc:
            best_acc = g_acc_valid.avg*100
            best_epoch = epoch
            best_acc_model = deepcopy(model)
        
        print(f"BEST VALID ACC: {best_acc:2.2f}")

        # print(f"[{epoch:2}/{config.n_epoch:2}] TRAIN LOSS: {loss_train.avg:.4f} TRAIN ACC: {acc_train.avg*100:2.2f} | VALID LOSS: {loss_valid.avg:.4f} | VALID ACC: {acc_valid.avg*100:2.2f} | BEST VALID ACC: {best_acc:.4f} |")
    print(f"\nSave the best acc model in epoch {best_epoch} : {best_acc}%")
    torch.save(best_acc_model.state_dict(), f"./{get_save_path()}/{version}_{best_acc:.2f}.pt")