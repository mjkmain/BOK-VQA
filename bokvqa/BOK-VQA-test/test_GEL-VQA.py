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
from types import SimpleNamespace

logging.set_verbosity_error()

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

args = get_test_arguments()

tokenizer = get_tokenizer()
train_transform, valid_transform = get_transform()
train_data, valid_data, triple_ans_list, triple_num_target, gold_ans_list, gold_num_target = get_data(args)
KGEModel, kg, emb_entity_, emb_rel_ = get_KGE(config, args.kge_data, args.kge_model)

test_data = pd.read_csv(f"./data/BOKVQA_data_test_{args.lang}.csv")
test_dataset = GELVQATestDataset(tokenizer, test_data, gold_ans_list, triple_ans_list, config.max_token, valid_transform, config, emb_entity_, emb_rel_, kg)

model = GELVQA(gold_num_target, triple_num_target)
model.load_state_dict(torch.load(f"saved_model/{args.file_name}"))
model = model.to(device)

model.eval()

count_correct = 0
for i in trange(len(test_dataset)):
    with torch.no_grad():
        data = test_dataset[i]
        imgs = data['image'].to(device)
        q_bert_ids = data['q_ids'].to(device)
        q_bert_mask = data['q_mask'].to(device)

        #--
        h_out, r_out, t_out = model(q_bert_ids.unsqueeze(0), q_bert_mask.unsqueeze(0), imgs.unsqueeze(0), emb=0, get_gold=False)
        h_pred = torch.argmax(h_out, dim=1)
        r_pred = torch.argmax(r_out, dim=1)
        t_pred = torch.argmax(t_out, dim=1)

        head = [triple_ans_list['h'][i] for i in h_pred]
        rel = [triple_ans_list['r'][i] for i in r_pred]
        tail = [triple_ans_list['t'][i] for i in t_pred]

        emb = get_embedded_vec(1, head, rel, tail, emb_entity_, emb_rel_, kg)
        emb = emb.to(config.device)
        #--

        gold_out, _, _, _ = model(q_bert_ids.unsqueeze(0), q_bert_mask.unsqueeze(0), imgs.unsqueeze(0), emb, get_gold=True)


        answer = data['answer']
        if answer in gold_ans_list:
            ans_idx = gold_ans_list.index(answer)
        else:
            ans_idx = None
            
        pred = torch.argmax(gold_out, dim=1)
        if pred.item() == ans_idx:
            count_correct += 1

print(f"acc : {(count_correct/len(test_data))*100:.2f}")