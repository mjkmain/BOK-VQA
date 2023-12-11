from torch.optim import Adam

from torchkge.utils import MarginLoss

import os
import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.optim import Adam
from tqdm import tqdm

from torchkge.models import ConvKBModel

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.data_structures import KnowledgeGraph
from types import SimpleNamespace
import pickle
import sys

def train_kge(iters=50000, lr=1e-4, batch_size=512, margin=0.5):

    config = {
        'emb_dim' : 256,
        'ent_emb_dim' : 256,
        'rel_emb_dim' : 256,
        'lr' : lr,
        'iter' : iters,
        'batch_size' : batch_size,
        'margin' : margin,
        'n_filter' : 3,
        'version' : 'all'
    }
    args = SimpleNamespace(**config)

    data = pd.read_csv(f"../data/{args.version}_triple.csv")

    triple_list = []
    for index in tqdm(range(len(data))):
        
        triple_dict = {}
        triple_dict["head"] = str(data['h'][index])
        triple_dict["relation"] = str(data['r'][index])
        triple_dict["tail"] = str(data['t'][index])
    
        triple_list.append(triple_dict)
    df = pd.DataFrame(triple_list).drop_duplicates().reset_index(drop=True)
    
    ratingmap = {rate : i for i , rate in enumerate(df['relation'])}
    entity_ori = np.concatenate([df['head'].values ,df['tail'].values])
    entitymap = {entity : i for i , entity in enumerate(entity_ori)}

    df2 = df.copy()
    df2.columns = ['from','rel','to']

    df2['rel'].map(lambda x : ratingmap.get(x))
    df2['from'].map(lambda x : entitymap.get(x))
    df2['to'].map(lambda x : entitymap.get(x))

    print(f"# unique relation : {df2['rel'].nunique()}")
    print(f"# unique head : {df2['from'].nunique()}")
    print(f"# unique tail : {df2['to'].nunique()}")

    print(f"num triple : {len(data)}")
    kg = KnowledgeGraph(df2)
    kg_train = kg

    model_convEx = ConvKBModel(args.emb_dim,
                               args.n_filter,
                            kg_train.n_ent,
                            kg_train.n_rel,
                            )
    criterion = MarginLoss(args.margin)

    if cuda.is_available():
        cuda.empty_cache()
        model_convEx.cuda()
        criterion.cuda()

    optimizer = Adam(model_convEx.parameters(), lr=args.lr, weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(kg_train)
    train_loader = DataLoader(kg_train, batch_size=args.batch_size)

    iter_count = 0
    model_convEx.train()

    while iter_count < iters:
        running_loss = 0.0
        for batch in train_loader:
            h, t, r = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            n_h, n_t = sampler.corrupt_batch(h, t, r)
            optimizer.zero_grad()

            pos, neg = model_convEx(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            score = model_convEx.scoring_function(h, t, r).mean().item()
            iter_count += 1
        sys.stdout.write(
            "\r[ITER : %d/%d] [TRAIN LOSS : %.5f] [SCORE : %.5f]"
            %(
                iter_count, iters, running_loss/len(train_loader), score
            )
        )
        sys.stdout.flush()

    model_convEx.normalize_parameters()
    kge_config =   {'emb_dim' : args.emb_dim, 
                    'n_ent' : kg_train.n_ent,
                    'n_rel' : kg_train.n_rel,
                    'iter' : iters,
                    'lr' : lr,
                    'batch_size' : batch_size,
                    'conv_size' : 3,
                    'margin' : margin,
                    }
    
    if not os.path.exists('./kge_save'):
        os.mkdir('./kge_save')
    
    with open(f'./kge_save/convkb_{iters}_{lr}_{batch_size}_{margin}_{args.version}_kg.pkl', 'wb') as f:
        pickle.dump(kg, f)
        
    with open(f'./kge_save/convkb_{iters}_{lr}_{batch_size}_{margin}_{args.version}_config.pkl', 'wb') as f:
        pickle.dump(kge_config, f)
        
    torch.save(model_convEx.state_dict(), f'./kge_save/convkb_{iters}_{lr}_{batch_size}_{margin}_{args.version}.pt')
    return kg, model_convEx

if __name__=='__main__':
    train_kge()
