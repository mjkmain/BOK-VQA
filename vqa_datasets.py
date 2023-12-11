import torch
from PIL import Image 
import numpy as np

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, gold_ans_list, max_token, transform, config):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.gold_ans_list = gold_ans_list        
        self.transform = transform
        self.config = config
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        question = self.data['question'][index]
        answer = self.data['answer'][index] 
        img_loc = self.data['img_path'][index] 
        
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                )
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        image = Image.open(img_loc).convert('RGB')  
        image = self.transform(image) 

        answer_idx = self.gold_ans_list.index(answer)

        return {'q_ids': torch.tensor(ids, dtype=torch.long), 
                'q_mask': torch.tensor(mask, dtype=torch.long),
                'answer': torch.tensor(answer_idx, dtype=torch.long),
                'image': image,}

class BaselineTestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, gold_ans_list, max_token, transform, config):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.gold_ans_list = gold_ans_list        
        self.transform = transform
        self.config = config
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        question = self.data['question'][index]
        answer = self.data['answer'][index] 
        img_loc = self.data['img_path'][index] 
        
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                )
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        image = Image.open(img_loc).convert('RGB')  
        image = self.transform(image) 


        return {'q_ids': torch.tensor(ids, dtype=torch.long), 
                'q_mask': torch.tensor(mask, dtype=torch.long),
                'answer': answer,
                'image': image,}
    
class GELVQAIdealDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, gold_ans_list, max_token, transform, config, *kge_args):
        
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.gold_ans_list = gold_ans_list        
        self.transform = transform
        self.config = config
        
        
        self.emb_entity_ = kge_args[0]
        self.emb_rel_ = kge_args[1]
        self.kg = kge_args[2]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        question = self.data['question'][index]
        answer = self.data['answer'][index] 
        
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                )
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']

        img_path = self.data['img_path'][index] 
        image = Image.open(img_path).convert('RGB')  
        image = self.transform(image) 
        
        answer_idx = self.gold_ans_list.index(answer)
        h, r, t = self.data['h'][index], self.data['r'][index], self.data['t'][index]
        h_emb = self.emb_entity_[self.kg.ent2ix[h]]
        r_emb = self.emb_rel_[self.kg.rel2ix[r]]
        t_emb = self.emb_entity_[self.kg.ent2ix[t]]

        return {
            'q_ids': torch.tensor(ids, dtype=torch.long), 
            'q_mask': torch.tensor(mask, dtype=torch.long),
            'image': image,
            'answer': torch.tensor(answer_idx, dtype=torch.long),
            'kge' : torch.tensor(np.concatenate([h_emb, r_emb, t_emb]), dtype=torch.float) 
            }

class GELVQAIdealTestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, gold_ans_list, max_token, transform, config, *kge_args):
        
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.gold_ans_list = gold_ans_list        
        self.transform = transform
        self.config = config
        
        
        self.emb_entity_ = kge_args[0]
        self.emb_rel_ = kge_args[1]
        self.kg = kge_args[2]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        question = self.data['question'][index]
        answer = self.data['answer'][index] 
        
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                )
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']

        img_path = self.data['img_path'][index] 
        image = Image.open(img_path).convert('RGB')  
        image = self.transform(image) 
        
        h, r, t = self.data['h'][index], self.data['r'][index], self.data['t'][index]
        h_emb = self.emb_entity_[self.kg.ent2ix[h]]
        r_emb = self.emb_rel_[self.kg.rel2ix[r]]
        t_emb = self.emb_entity_[self.kg.ent2ix[t]]

        return {
            'q_ids': torch.tensor(ids, dtype=torch.long), 
            'q_mask': torch.tensor(mask, dtype=torch.long),
            'image': image,
            'answer' : answer,
            'kge' : torch.tensor(np.concatenate([h_emb, r_emb, t_emb]), dtype=torch.float) 
            }
    
class GELVQADataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, gold_ans_list, triple_ans_list, max_token, transform, mode, config, *kge_args):
        
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.gold_ans_list = gold_ans_list        
        self.triple_ans_list = triple_ans_list
        self.transform = transform
        self.config = config
        self.mode = mode
        
        self.emb_entity_ = kge_args[0]
        self.emb_rel_ = kge_args[1]
        self.kg = kge_args[2]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        question = self.data['question'][index]
        answer = self.data['answer'][index] 
        img_loc = self.data['img_path'][index] 
        
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                )
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        image = Image.open(img_loc).convert('RGB')  
        image = self.transform(image) 
        
        answer_idx = self.gold_ans_list.index(answer)
        h, r, t = self.data['h'][index], self.data['r'][index], self.data['t'][index]
        h_label = self.triple_ans_list['h'].index(h)
        r_label = self.triple_ans_list['r'].index(r)
        t_label = self.triple_ans_list['t'].index(t)
        
        return {
            'q_ids': torch.tensor(ids, dtype=torch.long), 
            'q_mask': torch.tensor(mask, dtype=torch.long),
            'image': image,
            'answer': torch.tensor(answer_idx, dtype=torch.long),
            'h_label' : torch.tensor(h_label, dtype=torch.long),
            'r_label' : torch.tensor(r_label, dtype=torch.long),
            't_label' : torch.tensor(t_label, dtype=torch.long),
            }


class GELVQATestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, gold_ans_list, triple_ans_list, max_token, transform, config, *kge_args):
        
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.gold_ans_list = gold_ans_list        
        self.triple_ans_list = triple_ans_list
        self.transform = transform
        self.config = config

        self.emb_entity_ = kge_args[0]
        self.emb_rel_ = kge_args[1]
        self.kg = kge_args[2]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        question = self.data['question'][index]
        answer = self.data['answer'][index] 
        img_loc = self.data['img_path'][index] 
        
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                )
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        image = Image.open(img_loc).convert('RGB')  
        image = self.transform(image) 
            
        return {
            'q_ids': torch.tensor(ids, dtype=torch.long), 
            'q_mask': torch.tensor(mask, dtype=torch.long),
            'answer': answer,
            'image': image,
            }