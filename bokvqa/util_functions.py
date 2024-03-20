import torch 
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import torch
import pickle
import os
from torchkge.models import ConvKBModel
from torchkge.models import ComplExModel
from torchkge.models.translation import TransEModel, TransHModel, TransRModel, TransDModel,TorusEModel
from torchkge.models.bilinear import RESCALModel, DistMultModel, HolEModel, AnalogyModel

import torchvision.transforms as transforms
from transformers import AutoTokenizer
import transformers
import torchvision.models as models
import argparse
from datasets import load_dataset
import datasets

if __name__ == "__main__":
    print(os.getcwd())

# KGE_DIR = "/PATH/TO/SAVED-KGE"
KGE_DIR = "/home/nips/BOK-VQA/bokvqa/KGE-train"
# DATA_DIR = "/PATH/TO/DATA"
DATA_DIR = "/home/nips/BOK-VQA/bokvqa/data"

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_all_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def get_embedded_vec(batch_size, heads, rels, tails, emb_entity_ , emb_rel_, kg):
    head_emb = emb_entity_[[kg.ent2ix[i] for i in heads]]
    rel_emb = emb_rel_[[kg.rel2ix[i] for i in rels]]
    tail_emb = emb_entity_[[kg.ent2ix[i] for i in tails]]
    emb_list = []
    for i in range(batch_size):
        a = torch.tensor([head_emb[i], rel_emb[i], tail_emb[i]], dtype=torch.float).reshape(-1,768)
        emb_list.append(a)
    
    return torch.cat(emb_list, 0)
        
def save_results(version, train_loss, train_acc, valid_loss, valid_acc):
    PATH = f"./saved_results/{version}"
    with open(os.path.join(PATH, "train_loss.pkl"), 'wb') as f:
        pickle.dump(train_loss, f)

    with open(os.path.join(PATH, "train_acc.pkl"), 'wb') as f:
        pickle.dump(train_acc, f)

    with open(os.path.join(PATH, "valid_loss.pkl"), 'wb') as f:
        pickle.dump(valid_loss, f)

    with open(os.path.join(PATH, "valid_acc.pkl"), 'wb') as f:
        pickle.dump(valid_acc, f)

def save_config(version, config):
    PATH = f"./saved_results/{version}"
    with open(os.path.join(PATH, "config.txt"), "w") as f:
        for d in dir(config):
            if not d.startswith("__"):
                output = f"{d} : {getattr(config, d)}\n"
                f.write(output)

def get_KGE(config, kge_data='all', kge_model='convkb'):
    kge_dict = {
        'TransE' : TransEModel,
        'TransH' : TransHModel,
        'TransR' : TransRModel,
        'TransD' : TransDModel,
        'TorusE' : TorusEModel,
        'RESCAL' : RESCALModel,
        'DistMult' : DistMultModel,
        'HolE' : HolEModel,
        'complex' : ComplExModel,
        'convkb' : ConvKBModel,
        'Analogy' : AnalogyModel
    }

    with open(os.path.join(KGE_DIR, f'kge_save/{kge_model}_{config.kge_n_iter}_{config.kge_lr}_{config.kge_batch}_{config.kge_margin}_{kge_data}_config.pkl'), 'rb') as f:
        kge_config = pickle.load(f)

    with open(os.path.join(KGE_DIR, f'kge_save/{kge_model}_{config.kge_n_iter}_{config.kge_lr}_{config.kge_batch}_{config.kge_margin}_{kge_data}_kg.pkl'), 'rb') as f:
        kg = pickle.load(f)

    model = kge_dict[kge_model]

    if kge_model == 'convkb':
        KGEModel = model(kge_config['emb_dim'],
                                      3,
                                      kge_config['n_ent'],
                                      kge_config['n_rel']
                                    )
        
    elif kge_model in ["TransR", "TransD"]:
        KGEModel = model(      
                    kge_config['emb_dim'],
                    kge_config['emb_dim'],
                    kge_config['n_ent'],
                    kge_config['n_rel']
                )
    elif kge_model in ["TorusE"]:    
         KGEModel = model(      
                    kge_config['emb_dim'],
                    kge_config['n_ent'],
                    kge_config['n_rel'],
                    dissimilarity_type = 'torus_L2'
                    )
        
    else:
        KGEModel = model(kge_config['emb_dim'],
                         kge_config['n_ent'],
                         kge_config['n_rel']
                         )
      
    KGEModel.load_state_dict(torch.load(os.path.join(KGE_DIR, f'kge_save/{kge_model}_{config.kge_n_iter}_{config.kge_lr}_{config.kge_batch}_{config.kge_margin}_{kge_data}.pt')))
            
    emb_entity_ = KGEModel.get_embeddings()[0].detach().cpu().numpy()
    emb_rel_ = KGEModel.get_embeddings()[1].detach().cpu().numpy()

    return KGEModel, kg, emb_entity_, emb_rel_


def get_transform():

    train_transform = transforms.Compose(
        [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform



def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    return tokenizer

def get_data(args):
    if args.lang == 'bi':   
        data_ko = pd.read_csv(os.path.join(DATA_DIR, "BOKVQA_data_ko.csv"))
        data_en = pd.read_csv(os.path.join(DATA_DIR, "BOKVQA_data_en.csv"))
        data = pd.concat([data_ko, data_en])
    
    else:
        data = pd.read_csv(os.path.join(DATA_DIR, f'BOKVQA_data_{args.lang}.csv'))

    train_data = data[data[f'fold']!=1].reset_index(drop=True)
    valid_data = data[data[f'fold']==1].reset_index(drop=True)

    h_list = sorted(list(set(data['h'])))
    r_list = sorted(list(set(data['r'])))
    t_list = sorted(list(set(data['t'])))
    triple_ans_list = {"h":h_list, "r":r_list, "t":t_list}
    triple_target_num = {"h":len(h_list), "r":len(r_list), "t":len(t_list)}

    ans_list = sorted(list(pd.concat([train_data, valid_data], axis=0)['answer'].unique()))
    return train_data, valid_data, triple_ans_list, triple_target_num, ans_list, len(ans_list)

def get_num_workers():
    return 5

def get_batch_size():
    return 128

def get_seed():
    return 56

def get_language_model():
    return transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base', output_attentions=True)

def get_image_model():
    return models.resnet50(pretrained=True)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=['ko', 'en', 'bi'])
    parser.add_argument("--kge_data", type=str, required=False, default='all')
    parser.add_argument("--kge_model", type=str, required=False, choices=['complex', 'convkb','TransE', 'TransH',
                                                                          'TransR', 'TransD', 'TorusE', 'RESCAL',
                                                                          'DistMult', 'HolE', 'ComplExModel', 'Analogy'], default='convkb')

    return parser.parse_args()

def get_test_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=['ko', 'en', 'bi'])
    parser.add_argument("--kge_data", type=str, required=False, default='all')
    parser.add_argument("--kge_model", type=str, required=False, choices=['complex', 'convkb','TransE', 'TransH',
                                                                          'TransR', 'TransD', 'TorusE', 'RESCAL',
                                                                          'DistMult', 'HolE', 'ComplExModel', 'Analogy'], default='convkb')
    parser.add_argument("--file_name", type=str, required=True)
    return parser.parse_args()

def get_save_path():
    return "saved_model"
