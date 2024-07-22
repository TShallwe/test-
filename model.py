import torch
import torch.nn as nn
from torch_scatter import scatter
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np
import copy

class TimeConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(TimeConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(4, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(4)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, query_sub_idxs, batch_idxs, emb_rel, emb_time, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        embedded_all = torch.tanh(embedding)
        e1_embedded = embedded_all[query_sub_idxs][batch_idxs].unsqueeze(1)
        r_embedded = emb_rel[batch_idxs].unsqueeze(1)
        batch_size = len(embedding)
        
        emb_time_1, emb_time_2 = emb_time
        emb_time_1 = emb_time_1[batch_idxs].unsqueeze(1)
        emb_time_2 = emb_time_2[batch_idxs].unsqueeze(1)

        stacked_inputs = torch.cat([e1_embedded, r_embedded, emb_time_1, emb_time_2], 1)  # batch_size,2,h_dim
        stacked_inputs = self.bn0(stacked_inputs)  # batch_size,2,h_dim
        x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim
        x = self.conv1(x)  # batch_size,2,h_dim
        x = self.bn1(x)  # batch_size,channels,h_dim
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)  # batch_size,channels*h_dim
        x = self.fc(x)  # batch_size,channels*h_dim
        x = self.hidden_drop(x)  # batch_size,h_dim
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.sum(x * embedded_all, dim=-1)
        else:
            x = torch.mm(x, embedded_all.transpose(1, 0))
            x = torch.mul(x, partial_embeding)
        return x

    def forward_slow(self, embedding, emb_rel, triplets):

        e1_embedded_all = torch.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        e2_embedded = e1_embedded_all[triplets[:, 2]]
        score = torch.sum(torch.mul(x, e2_embedded), dim=1)
        pred = score
        return pred
    

    def forward_time(self, embedding, query_sub_idxs, query_obj_idxs, batch_idxs, emb_rel, emb_time, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = torch.tanh(embedding)
        batch_size = len(embedding)
        e1_embedded = e1_embedded_all[query_sub_idxs].unsqueeze(1)  # batch_size,1,h_dim
        e2_embedded = e1_embedded_all[query_obj_idxs].unsqueeze(1)
        rel_embedded = emb_rel.unsqueeze(1)  # batch_size,1,h_dim
        scores = []
        time_size = 64
        triple_size = 64

        emb_time_1, emb_time_2 = emb_time[0].unsqueeze(1), emb_time[1].unsqueeze(1)

        extended_e1_embedded = e1_embedded.unsqueeze(1).repeat(1, emb_time_1.size(0), 1, 1) 
        extended_rel_embedded = rel_embedded.unsqueeze(1).repeat(1, emb_time_1.size(0), 1, 1) 
        batch_size = extended_e1_embedded.size(0)

        
        extended_time1 = emb_time_1.unsqueeze(0).repeat(extended_e1_embedded.size(0), 1, 1, 1) 
        extended_time2 = emb_time_2.unsqueeze(0).repeat(extended_e1_embedded.size(0), 1, 1, 1) 
        
        stacked_inputs = torch.cat([extended_e1_embedded, extended_rel_embedded, extended_time1, extended_time2], 2).view(-1,4,extended_time1.size(-1))  # batch_size,2,h_dim
        
        stacked_inputs = self.bn0(stacked_inputs)  # batch_size,2,h_dim
        x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim

        del extended_time1, extended_time2, extended_e1_embedded, extended_rel_embedded, stacked_inputs

        x = self.conv1(x)  # batch_size,2,h_dim
        x = self.bn1(x)  # batch_size,channels,h_dim
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size*emb_time_1.size(0), -1)  # batch_size,channels*h_dim
        x = self.fc(x)  # batch_size,channels*h_dim
        x = self.hidden_drop(x)  # batch_size,h_dim
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, -1, x.size(1))
        if partial_embeding is None:
            x = torch.sum(x * e2_embedded, axis=-1)
       
     

       
        scores = torch.log(torch.softmax(x, axis=1))
        return scores

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, selfloop=False, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        # self.selfloop = selfloop
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        # if self.selfloop:
        #     self.self_loop_transform = nn.Linear(out_dim, out_dim)  # 自环转换层

    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
       
        hs = hidden[sub]
        hr = self.rela_embed(rel) # relation embedding of each edge
       
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation
        
        # message aggregation
        message = hs * hr 
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # if self.selfloop:
        #     obj_hidden = hidden[obj]
        #     message_agg[obj] += self.act(self.self_loop_transform(obj_hidden))
            
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))
        
        if shortcut: hidden_new = hidden_new + hidden
        
        return hidden_new

class TemporalGNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_time, act=lambda x:x):
        super(TemporalGNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.time_embed = nn.Embedding(n_time, in_dim)  # time embedding
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.Wt_attn = nn.Linear(in_dim, attn_dim, bias=False)  # time attention
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        times = edges[:, 3]
        hs = hidden[sub]
        hr = self.rela_embed(rel) # relation embedding of each edge
        ht = self.time_embed(times) # time embedding of each edge
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation
        
        # message aggregation
        message = hs * hr * ht
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr) + self.Wt_attn(ht))))
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))
        
        if shortcut: hidden_new = hidden_new + hidden
        
        return hidden_new

class TemporalGNNLayerv2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_time, act=lambda x:x):
        super(TemporalGNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.time_embed = nn.Embedding(n_time, in_dim)  # time embedding
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        times = edges[:, 3]
        hs = hidden[sub] 
        hr = self.rela_embed(rel) # relation embedding of each edge
        ht = self.time_embed(times) # time embedding of each edge
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation
        
        # message aggregation
        message = hs * hr + ht
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))
        
        if shortcut: hidden_new = hidden_new + hidden
        
        return hidden_new

class GNN_auto(torch.nn.Module):
    def __init__(self, params, loader, selfloop=False):
        super(GNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        self.device = torch.device('cpu') if params.gpu == -1 else torch.device('cuda')
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, selfloop, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        
    def forward(self, q_sub, q_rel, times, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, query_obj_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).to(self.device)
        hidden = torch.zeros(n_node, self.hidden_dim).to(self.device)
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).to(self.device)
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all
    

class TemporalGNN_auto(torch.nn.Module):
    def __init__(self, params, loader, n_time):
        super(TemporalGNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        self.device = torch.device('cpu') if params.gpu == -1 else torch.device('cuda')
        self.n_time = n_time
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(TemporalGNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, loader.n_time, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        
    def forward(self, q_sub, q_rel, q_tim, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).to(self.device)
        hidden = torch.zeros(n_node, self.hidden_dim).to(self.device)
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).to(self.device)
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all
    
class TemporalGNN_autov2(torch.nn.Module):
    def __init__(self, params, loader, n_time):
        super(TemporalGNN_autov2, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        
        self.loader = loader
        self.device = torch.device('cpu') if params.gpu == -1 else torch.device('cuda:2')
        self.n_time = n_time
        self.all_timestamps = torch.LongTensor(range(self.n_time+1)).to(self.device)
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.sin = torch.sin
        self.decoder = TimeConvTransE(self.n_rel, self.hidden_dim)
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
    
    def get_init_time(self, q_tim):
        
        q_tim = q_tim.unsqueeze(1).float()
        t1 = self.weight_t1 * q_tim + self.bias_t1
        t2 = self.sin(self.weight_t2 * q_tim + self.bias_t2)
        return t1, t2
    
    def forward(self, q_sub, q_rel, q_tim, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, query_obj_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).to(self.device)
        hidden = torch.zeros(n_node, self.hidden_dim).to(self.device)
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

    
    
        time_embs = self.get_init_time(q_tim)
        scores = self.decoder.forward(hidden, query_sub_idxs, batch_idxs, self.query_rela_embed(q_rel), time_embs)
        time_loss = 0
        if mode == 'train':
            all_time_embs = self.get_init_time(self.all_timestamps)
            time_score = self.decoder.forward_time(hidden, query_sub_idxs, query_obj_idxs, batch_idxs, self.query_rela_embed(q_rel), all_time_embs)
            time_loss = F.nll_loss(time_score, q_tim)
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).to(self.device)
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all, time_loss

class TemporalNonlinear(torch.nn.Module):
    def __init__(self, params, loader, n_time):
        super(TemporalNonlinear, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        self.device = torch.device('cpu') if params.gpu == -1 else torch.device('cuda')
        self.n_time = n_time
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.nonlinear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)
        
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, self.hidden_dim))
        self.sin = torch.sin
        self.decoder = TimeConvTransE(self.n_rel, self.hidden_dim)
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
    
    def get_init_time(self, q_tim):
        
        q_tim = q_tim.unsqueeze(1).float()
        t1 = self.weight_t1 * q_tim + self.bias_t1
        t2 = self.sin(self.weight_t2 * q_tim + self.bias_t2)
        return t1, t2
    
    def forward(self, q_sub, q_rel, q_tim, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).to(self.device)
        hidden = torch.zeros(n_node, self.hidden_dim).to(self.device)
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        hidden = self.nonlinear(hidden)
        hidden = self.bn(hidden)
        hidden = self.dropout(hidden)

        # readout
        time_embs = self.get_init_time(q_tim)
        scores = self.decoder.forward(hidden, query_sub_idxs, batch_idxs, self.query_rela_embed(q_rel), time_embs)
        
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).to(self.device)
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all
      

class CENET(nn.Module):
    def __init__(self, num_e, num_rel, num_t, args):
        super(CENET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args

        # entity relation embedding
        self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel, args.embedding_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, args.embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.linear_frequency = nn.Linear(self.num_e, args.embedding_dim)

        self.contrastive_hidden_layer = nn.Linear(3 * args.embedding_dim, args.embedding_dim)
        self.contrastive_output_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.oracle_layer = Oracle(3 * args.embedding_dim, 1)
        self.oracle_layer.apply(self.weights_init)

        self.linear_pred_layer_s1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.weights_init(self.linear_frequency)
        self.weights_init(self.linear_pred_layer_s1)
        self.weights_init(self.linear_pred_layer_o1)
        self.weights_init(self.linear_pred_layer_s2)
        self.weights_init(self.linear_pred_layer_o2)

        """
        pe = torch.zeros(400, 3 * args.embedding_dim)
        position = torch.arange(0, 400, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, 3 * args.embedding_dim, 2).float() * (-math.log(10000.0) / (3 * args.embedding_dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        """

        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()
        self.oracle_mode = args.oracle_mode

        print('CENET Initiated')

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_block, mode_lk, total_data=None):
        quadruples, s_history_event_o, o_history_event_s, \
        s_history_label_true, o_history_label_true, s_frequency, o_frequency = batch_block

        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            sub_rank, obj_rank, batch_loss = [None] * 3
            if mode_lk == 'Training':
                return batch_loss
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, batch_loss
            else:
                return None

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        """
        t = (quadruples[:, 3] / 24.0).long()
        time_embedding = self.pe[t]
        """

        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = self.args.lambdax
        o_history_tag[o_history_tag != 0] = self.args.lambdax

        s_non_history_tag[s_history_tag == 1] = -self.args.lambdax
        s_non_history_tag[s_history_tag == 0] = self.args.lambdax

        o_non_history_tag[o_history_tag == 1] = -self.args.lambdax
        o_non_history_tag[o_history_tag == 0] = self.args.lambdax

        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax

        s_frequency = F.softmax(s_frequency, dim=1)
        o_frequency = F.softmax(o_frequency, dim=1)
        s_frequency_hidden = self.tanh(self.linear_frequency(s_frequency))
        o_frequency_hidden = self.tanh(self.linear_frequency(o_frequency))

        if mode_lk == 'Training':
            s_nce_loss, _ = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                    s_history_tag, s_non_history_tag)
            o_nce_loss, _ = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                    o_history_tag, o_non_history_tag)
            # calculate_spc_loss(self, hidden_lk, actor1, r, rel_embeds, targets):
            s_spc_loss = self.calculate_spc_loss(s, r, self.rel_embeds[:self.num_rel],
                                                 s_history_label_true, s_frequency_hidden)
            o_spc_loss = self.calculate_spc_loss(o, r, self.rel_embeds[self.num_rel:],
                                                 o_history_label_true, o_frequency_hidden)
            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            spc_loss = (s_spc_loss + o_spc_loss) / 2.0
            # print('nce loss', nce_loss.item(), ' spc loss', spc_loss.item())
            return self.args.alpha * nce_loss + (1 - self.args.alpha) * spc_loss

        elif mode_lk in ['Valid', 'Test']:
            s_history_oid = []
            o_history_sid = []

            for i in range(quadruples.shape[0]):
                s_history_oid.append([])
                o_history_sid.append([])
                for con_events in s_history_event_o[i]:
                    s_history_oid[-1] += con_events[:, 1].tolist()
                for con_events in o_history_event_s[i]:
                    o_history_sid[-1] += con_events[:, 1].tolist()

            s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                          self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                          s_history_tag, s_non_history_tag)
            o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                          self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                          o_history_tag, o_non_history_tag)

            s_ce_loss, s_pred_history_label, s_ce_all_acc = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                                                             s_history_label_true, s_frequency_hidden)
            o_ce_loss, o_pred_history_label, o_ce_all_acc = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                                                             o_history_label_true, o_frequency_hidden)

            s_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))

            for i in range(quadruples.shape[0]):
                if s_pred_history_label[i].item() > 0.5:
                    s_mask[i, s_history_oid[i]] = 1
                else:
                    s_mask[i, :] = 1
                    s_mask[i, s_history_oid[i]] = 0

                if o_pred_history_label[i].item() > 0.5:
                    o_mask[i, o_history_sid[i]] = 1
                else:
                    o_mask[i, :] = 1
                    o_mask[i, o_history_sid[i]] = 0

            if self.oracle_mode == 'soft':
                s_mask = F.softmax(s_mask, dim=1)
                o_mask = F.softmax(o_mask, dim=1)


            s_total_loss1, sub_rank1 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask, total_data, 's', True)
            o_total_loss1, obj_rank1 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask, total_data, 'o', True)
            batch_loss1 = (s_total_loss1 + o_total_loss1) / 2.0

            s_total_loss2, sub_rank2 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask, total_data, 's', False)
            o_total_loss2, obj_rank2 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask, total_data, 'o', False)
            batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0

            # Ground Truth
            s_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))


            for i in range(quadruples.shape[0]):
                if o[i] in s_history_oid[i]:
                    s_mask_gt[i, s_history_oid[i]] = 1
                else:
                    s_mask_gt[i, :] = 1
                    s_mask_gt[i, s_history_oid[i]] = 0

                if s[i] in o_history_sid[i]:
                    o_mask_gt[i, o_history_sid[i]] = 1
                else:
                    o_mask_gt[i, :] = 1
                    o_mask_gt[i, o_history_sid[i]] = 0

            s_total_loss3, sub_rank3 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask_gt, total_data, 's', True)
            o_total_loss3, obj_rank3 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask_gt, total_data, 'o', True)
            batch_loss3 = (s_total_loss3 + o_total_loss3) / 2.0

            return sub_rank1, obj_rank1, batch_loss1, \
                   sub_rank2, obj_rank2, batch_loss2, \
                   sub_rank3, obj_rank3, batch_loss3, \
                   (s_ce_all_acc + o_ce_all_acc) / 2

        elif mode_lk == 'Oracle':
            print('Oracle Training')
            s_ce_loss, _, _ = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                               s_history_label_true, s_frequency_hidden)
            o_ce_loss, _, _ = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                               o_history_label_true, o_frequency_hidden)
            return (s_ce_loss + o_ce_loss) / 2.0 + self.oracle_l1(0.01)

    def oracle_loss(self, actor1, r, rel_embeds, history_label, frequency_hidden):
        history_label_pred = F.sigmoid(
            self.oracle_layer(torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1)))
        tmp_label = torch.squeeze(history_label_pred).clone().detach()
        tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        # print('# Bias Ratio', torch.sum(tmp_label).item() / tmp_label.shape[0])
        ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        print('# CE Accuracy', ce_accuracy)
        ce_loss = self.crossEntropy(torch.squeeze(history_label_pred), torch.squeeze(history_label))
        return ce_loss, history_label_pred, ce_accuracy * tmp_label.shape[0]

    def calculate_nce_loss(self, actor1, actor2, r, rel_embeds, linear1, linear2, history_tag, non_history_tag):
        preds_raw1 = self.tanh(linear1(
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds1 = F.softmax(preds_raw1.mm(self.entity_embeds.transpose(0, 1)) + history_tag, dim=1)

        preds_raw2 = self.tanh(linear2(
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds2 = F.softmax(preds_raw2.mm(self.entity_embeds.transpose(0, 1)) + non_history_tag, dim=1)

        # cro_entr_loss = self.criterion_link(preds1 + preds2, actor2)

        nce = torch.sum(torch.gather(torch.log(preds1 + preds2), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]

        pred_actor2 = torch.argmax(preds1 + preds2, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print('# Batch accuracy', accuracy)

        return nce, preds1 + preds2

    def link_predict(self, nce_loss, preds, ce_loss, actor1, actor2, r, trust_musk, all_triples, pred_known, oracle,
                     history_tag=None, case_study=False):
        if case_study:
            # f = open("case_study.txt", "a+")
            # entity2id, relation2id = get_entity_relation_set(self.args.dataset)
            pass

        if oracle:
            preds = torch.mul(preds, trust_musk)
            print('$Batch After Oracle accuracy:', end=' ')
        else:
            print('$Batch No Oracle accuracy:', end=' ')
        # compute the correct triples
        pred_actor2 = torch.argmax(preds, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print(accuracy)
        # print('Batch Error', 1 - accuracy)

        total_loss = nce_loss + ce_loss

        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]
            if case_study:
                in_history = torch.where(history_tag[i] > 0)[0]
                not_in_history = torch.where(history_tag[i] < 0)[0]
                print('---------------------------', file=f)
                for hh in range(in_history.shape[0]):
                    print('his:', entity2id[in_history[hh].item()], file=f)

                print(pred_known,
                      'Truth:', entity2id[cur_s.item()], '--', relation2id[cur_r.item()], '--', entity2id[cur_o.item()],
                      'Prediction:', entity2id[pred_actor2[i].item()], file=f)

            o_label = cur_o
            ground = preds[i, cur_o].clone().item()
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 0]

                preds[i, idx] = 0
                preds[i, o_label] = ground

            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
        return total_loss, ranks

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.rel_embeds.pow(2)) + torch.mean(self.entity_embeds.pow(2))
        return regularization_loss * reg_param

    def oracle_l1(self, reg_param):
        reg = 0
        for param in self.oracle_layer.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param

    # contrastive
    def freeze_parameter(self):
        self.rel_embeds.requires_grad_(False)
        self.entity_embeds.requires_grad_(False)
        self.linear_pred_layer_s1.requires_grad_(False)
        self.linear_pred_layer_o1.requires_grad_(False)
        self.linear_pred_layer_s2.requires_grad_(False)
        self.linear_pred_layer_o2.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.contrastive_hidden_layer.requires_grad_(False)
        self.contrastive_output_layer.requires_grad_(False)

    def contrastive_layer(self, x):
        # Implement from the encoder E to the projection network P
        # x = F.normalize(x, dim=1)
        x = self.contrastive_hidden_layer(x)
        # x = F.relu(x)
        # x = self.contrastive_output_layer(x)
        # Normalize to unit hypersphere
        # x = F.normalize(x, dim=1)
        return x

    def calculate_spc_loss(self, actor1, r, rel_embeds, targets, frequency_hidden):
        projections = self.contrastive_layer(
            torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1))
        targets = torch.squeeze(targets)
        """if np.random.randint(0, 10) < 1 and torch.sum(targets) / targets.shape[0] < 0.65 and torch.sum(targets) / targets.shape[0] > 0.35:
            np.savetxt("xx.tsv", projections.detach().cpu().numpy(), delimiter="\t")
            np.savetxt("yy.tsv", targets.detach().cpu().numpy(), delimiter="\t")
        """
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = to_device(targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets)
        mask_anchor_out = to_device(1 - torch.eye(exp_dot_tempered.shape[0]))
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return 0
        return supervised_contrastive_loss
