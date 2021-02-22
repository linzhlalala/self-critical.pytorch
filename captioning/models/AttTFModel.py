from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import math

from .CaptionModel import CaptionModel
from .AttModel import AttModel,sort_pack_padded_sequence,pad_unsort_packed_sequence,pack_wrapper

from .FCModel import LSTMCore
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    #Mh Attention should not be change, make change at the forward of model 
    def __init__(self, h=8, d_model=512, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # s = q*k/sqrt(d)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # p = softmax(s)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # out = p*v
    return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MhAtt2in2Core(nn.Module):
    def __init__(self, opt):
        super(MhAtt2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #also also as d_model
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        #self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, self.rnn_size//2) #remove 5*
        self.h2h = nn.Linear(self.rnn_size, self.rnn_size//2) #remove 5*
        self.de2cell = nn.Linear(self.rnn_size, 4*self.rnn_size) #put 5* after tf
        self.dropout = nn.Dropout(self.drop_prob_lm)

        #TF component
        self.head = getattr(opt,'num_head',8)
        self.dropout_rate = getattr(opt,'drop_out',0.1)
        self.layer = getattr(opt,'num_layers',3)
        self.d_ff = getattr(opt,'d_ff',2048)
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.head, opt.rnn_size, self.dropout_rate)
        ff = PositionwiseFeedForward(opt.rnn_size, self.d_ff, self.dropout_rate)
        #att encoder go attmodel and call once for a seq
        self.att_encoder = Encoder(EncoderLayer(opt.rnn_size, c(attn), c(ff), self.dropout_rate), self.layer)
        for p in self.att_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        #word decoder to run decode before LSTM
        self.in_decoder = Decoder(DecoderLayer(opt.rnn_size, c(attn), c(attn), c(ff), self.dropout_rate), self.layer)
        for p in self.in_decoder .parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        #hx + hh
        all_input_sums = torch.cat((self.i2h(xt),self.h2h(state[0][-1])),1)
        
        #p_att_feats now stand for att encode
        #print("p_att_feats:",p_att_feats.shape)    
        #print("all_input_sums before:",all_input_sums.shape)     
        
        decodes = self.in_decoder(all_input_sums.unsqueeze(1), p_att_feats).squeeze() #maskS  = NONE
        #print("decodes:",decodes.shape)  

        all_input_sums = self.de2cell(decodes)
        #print("all_input_sums after:",all_input_sums.shape)  
        #lstm 
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(all_input_sums)
        in_gate =       sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate =   sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate =      sigmoid_chunk.narrow(1, 2*self.rnn_size, self.rnn_size)
        in_transform =  all_input_sums.narrow(1, 3*self.rnn_size, self.rnn_size)
        #in_transform = torch.max(\
        #    in_transform.narrow(1, 0, self.rnn_size),
        #    in_transform.narrow(1, self.rnn_size, self.rnn_size))
        #out
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class MhAtt2in2Model(AttModel):
    def __init__(self, opt):
        super(MhAtt2in2Model, self).__init__(opt)
        self.rnn_size = opt.rnn_size
        del self.embed, self.fc_embed, self.att_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = self.att_embed = lambda x: x

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        
        
        self.core = MhAtt2in2Core(opt)
        del self.ctx2att        
        self.ctx2att = self.core.att_encoder
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # encoder the att as a cache
        p_att_feats = self.ctx2att(att_feats,att_masks)

        return fc_feats, att_feats, p_att_feats, att_masks