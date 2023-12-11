from util_functions import *
from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, num_target_gold, dim_i=768, dim_h=1024):
        super(BaselineModel, self).__init__()
        self.bert = get_language_model()
        self.bert.pooler.activation = nn.Identity()

        self.i_model = get_image_model()
        self.i_model.fc = nn.Linear(self.i_model.fc.in_features, dim_i)
        self.linear = nn.Linear(dim_i, dim_h)
        
        self.relu =  nn.ReLU()
        self.h_layer_norm = nn.LayerNorm(dim_h)

        self.linear_gold = nn.Linear(dim_h, num_target_gold)
        self.drop = nn.Dropout(0.2)

    def forward(self, idx, mask, image):
        q_f = self.bert(idx, mask) 
        q_f = q_f.pooler_output
        i_f = self.i_model(image)
        i_f = self.drop(i_f)
        
        iq_f = i_f * q_f

        out_iq = self.drop(self.relu(self.h_layer_norm(self.linear(iq_f))))
        out_gold = self.linear_gold(out_iq)

        return out_gold

class GELVQAIdeal(nn.Module):
    def __init__(self, num_target_gold, dim_i=768, dim_h=1024):
        super(GELVQAIdeal, self).__init__()
        self.bert = get_language_model()
        self.bert.pooler.activation = nn.Identity()

        self.i_model = get_image_model()
        self.i_model.fc = nn.Linear(self.i_model.fc.in_features, dim_i)
        self.linear = nn.Linear(768, dim_h)
        
        self.relu = nn.ReLU()
        self.h_layer_norm = nn.LayerNorm(dim_h)
        
        self.linear_gold = nn.Linear(dim_h, num_target_gold)
        self.drop = nn.Dropout(0.2)
        
    def forward(self, idx, mask, image, emb):
        q_f = self.bert(idx, mask) 
        q_f = q_f[0][:, 0, :]
        i_f = self.i_model(image)
        i_f = self.drop(i_f)

        iq_f = i_f*q_f 

        embed = emb.squeeze()
        uni_f = embed*iq_f
        out_uni = self.drop(self.relu(self.h_layer_norm(self.linear(uni_f))))
        out_gold = self.linear_gold(out_uni)
        
        return out_gold
    
class GELVQA(nn.Module):
    def __init__(self, num_target_gold, num_target_triple, dim_i=768, dim_h=1024):
        super(GELVQA, self).__init__()
        
        self.bert = get_language_model()
        self.bert.pooler.activation = nn.Identity()

        self.i_model = get_image_model()
        self.i_model.fc = nn.Linear(self.i_model.fc.in_features, dim_i)
        self.linear = nn.Linear(768, dim_h)
        
        self.relu = nn.ReLU()
        self.h_layer_norm = nn.LayerNorm(dim_h)
        
        self.linear_gold = nn.Linear(dim_h, num_target_gold)
        self.linear_h = nn.Linear(dim_h, num_target_triple['h'])
        self.linear_r = nn.Linear(dim_h, num_target_triple['r'])
        self.linear_t = nn.Linear(dim_h, num_target_triple['t'])
        self.drop = nn.Dropout(0.2)

    def forward(self, idx, mask, image, emb=0, get_gold=True):
        q_f = self.bert(idx, mask) 
        q_f = q_f[0][:, 0, :]
        i_f = self.i_model(image)
        i_f = self.drop(i_f)

        iq_f = i_f*q_f

        out_q = self.drop(self.relu(self.h_layer_norm(self.linear(q_f))))
        out_iq = self.drop(self.relu(self.h_layer_norm(self.linear(iq_f))))
        out_h = self.linear_h(out_iq)
        out_r = self.linear_r(out_q)
        out_t = self.linear_t(out_q)

        if get_gold:
            embed = emb.squeeze()
            uni_f = embed*iq_f
            out_uni = self.drop(self.relu(self.h_layer_norm(self.linear(uni_f))))
            out_gold = self.linear_gold(out_uni)
            return out_gold, out_h, out_r, out_t
        
        if not get_gold:
            return out_h, out_r, out_t
        
class GELVQAAttn(nn.Module):
    def __init__(self, num_target_gold, num_target_triple, dim_i=768, dim_h=1024):
        super(GELVQAAttn, self).__init__()
        
        self.bert = get_language_model()
        self.bert.pooler.activation = nn.Identity()

        self.i_model = get_image_model()
        self.i_model.fc = nn.Linear(self.i_model.fc.in_features, dim_i)
        self.linear = nn.Linear(768, dim_h)
        
        self.relu = nn.ReLU()
        self.h_layer_norm = nn.LayerNorm(dim_h)
        
        self.linear_gold = nn.Linear(dim_h, num_target_gold)
        self.linear_h = nn.Linear(dim_h, num_target_triple['h'])
        self.linear_r = nn.Linear(dim_h, num_target_triple['r'])
        self.linear_t = nn.Linear(dim_h, num_target_triple['t'])

        self.kge_linear_1 = nn.Linear(768, 768)
        self.kge_linear_2 = nn.Linear(768, 768)

        self.drop = nn.Dropout(0.2)
        self.mha = nn.MultiheadAttention(256, 1, batch_first=True)
        
    def forward(self, idx, mask, image, emb=0, get_gold=True, bs=get_batch_size()):
        q_f = self.bert(idx, mask)

        attention_score = q_f[-1]
        q_f = q_f[0][:, 0, :]

        i_f = self.i_model(image)
        i_f = self.drop(i_f)

        iq_f = i_f*q_f 

        out_q = self.drop(self.relu(self.h_layer_norm(self.linear(q_f))))
        out_iq = self.drop(self.relu(self.h_layer_norm(self.linear(iq_f))))
        out_h = self.linear_h(out_iq)
        out_r = self.linear_r(out_q)
        out_t = self.linear_t(out_q)


        if get_gold:
            embed = emb.squeeze()
            embed = embed.reshape((bs, 3, 256))
            emb_attn_output, emb_attn_output_weight = self.mha(embed, embed, embed)
            emb_attn_output = emb_attn_output.reshape(bs, -1)

            embed = self.kge_linear_2(self.relu(self.kge_linear_1(emb_attn_output)))

            uni_f = embed*iq_f
            out_uni = self.drop(self.relu(self.h_layer_norm(self.linear(uni_f))))
            out_gold = self.linear_gold(out_uni)
            return out_gold, out_h, out_r, out_t, attention_score, emb_attn_output_weight
        
        if not get_gold:
            return out_h, out_r, out_t, attention_score