# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
        init_alpha: float
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.register_parameter('alpha', nn.Parameter(torch.tensor(init_alpha)))
    
    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self, 
        tgt, 
        s_memory,
        s_pos=None,
        l_memory=None,
        l_pos=None,
        query_pos=None,
        num_k_exclude_rope: int = 0,
    ):

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt_s = self._forward_ca(tgt, s_memory, query_pos, s_pos, num_k_exclude_rope=num_k_exclude_rope)
        
        if l_memory is not None and l_memory.shape[1] > 0:
            tgt_l =  self._forward_ca(tgt, l_memory, query_pos, l_pos, num_k_exclude_rope=num_k_exclude_rope)
            tgt    = tgt_s + torch.tanh(self.alpha) * tgt_l
        else:
            tgt = tgt_s
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
    
    def forward(
        self,
        curr : torch.Tensor,                       # self-attention inputs; list[tensor] or tensor
        curr_pos=None,              # pos_enc for self-attention inputs
        s_memory=None,              # SHORT bank (with pointers appended at tail)
        s_pos=None,
        l_memory=None,              # LONG bank (can be None/empty)
        l_pos=None,
        num_obj_ptr_tokens: int = 0,    # number of pointer tokens at tail of each bank
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list) and len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]
        # shapes check for batch dim
        if s_memory is not None:
            assert curr.shape[1] == s_memory.shape[1], "Batch size mismatch (curr vs s_memory)"
        if l_memory is not None and l_memory.shape[1] > 0:
            assert curr.shape[1] == l_memory.shape[1], "Batch size mismatch (curr vs l_memory)"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # Convert to batch-first for the layers if needed
        if self.batch_first:
            output   = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1) if curr_pos is not None else None
            s_memory = s_memory.transpose(0, 1) if s_memory is not None else None
            s_pos    = s_pos.transpose(0, 1) if s_pos is not None else None
            if l_memory is not None:
                l_memory = l_memory.transpose(0, 1)
            if l_pos is not None:
                l_pos = l_pos.transpose(0, 1)

        # Pass through N identical layers
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                s_memory=s_memory,
                s_pos=s_pos,
                l_memory=l_memory,
                l_pos=l_pos,
                query_pos=curr_pos,
                **kwds,
            )

        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output

