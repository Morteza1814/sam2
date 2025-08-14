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
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class DualMemoryAttentionLayer(MemoryAttentionLayer):
    def __init__(self, init_alpha: float, *args, **kw):
        super().__init__(*args, **kw)
        self.register_parameter('alpha', nn.Parameter(torch.tensor(init_alpha)))

    def forward(self, tgt, memory, pos=None, query_pos=None,
                s_bank_len: int = 0,          #  NEW
                num_k_exclude_rope: int = 0,
                cond_tokens: int = 0,
                noncond_tokens: int = 0):

        # 1) Self-attention (unchanged)
        tgt = self._forward_sa(tgt, query_pos)

        seq_dim = 1 # if self.batch_first else 0
        total_len = memory.shape[seq_dim]
        ptr_start = total_len - num_k_exclude_rope     
        noncond_start = cond_tokens 
        assert noncond_start + noncond_tokens == ptr_start, "counts mismatch"

        # --- compute how many non-cond tokens to include in S ---
        k_nc = max(0, min(noncond_tokens, s_bank_len - cond_tokens))

        if seq_dim:
            # S = [all cond] + [last k_nc of non-cond]
            s_bank_left  = memory[:, :cond_tokens, :]
            s_bank_right = memory[:, ptr_start - k_nc:ptr_start, :] if k_nc > 0 else memory[:, :0, :]
            s_bank = torch.cat([s_bank_left, s_bank_right], dim=seq_dim)

            # L = remaining non-cond (the far part we didn't include)
            l_bank = memory[:, noncond_start: ptr_start - k_nc, :] if (noncond_tokens - k_nc) > 0 else memory[:, :0, :]

            # Pointers
            p_bank = memory[:, ptr_start:, :]

            # Positional encodings match the same slices
            s_pos_left  = pos[:, :cond_tokens, :] if pos is not None else None
            s_pos_right = (pos[:, ptr_start - k_nc:ptr_start, :] if (pos is not None and k_nc > 0) else None)
            s_pos = None if pos is None else (s_pos_left if k_nc == 0 else torch.cat([s_pos_left, s_pos_right], dim=seq_dim))

            l_pos = (pos[:, noncond_start: ptr_start - k_nc, :] if (pos is not None and (noncond_tokens - k_nc) > 0) else None)
            p_pos = pos[:, ptr_start:, :] if pos is not None else None
        else:
            raise NotImplementedError("seq_dim==0 path not used with batch_first=True")

        # --- build final K/V for S and optional L (always append pointers) ---
        s_mem  = torch.cat([s_bank, p_bank], dim=seq_dim)
        s_pos_ = None if s_pos is None else torch.cat([s_pos, p_pos], dim=seq_dim)

        # 2) α==0 → ONLY short bank (match the 7-frame baseline behavior)
        # if torch.allclose(torch.tanh(self.alpha.detach()), torch.zeros((), device=self.alpha.device)):
        #     tgt_s = self._forward_ca(tgt, s_mem, query_pos, s_pos_, num_k_exclude_rope)
        #     tgt2 = self.norm3(tgt_s)
        #     tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        #     return tgt_s + self.dropout3(tgt2)

        # 3) Otherwise, compute both branches and blend
        tgt_s = self._forward_ca(tgt, s_mem, query_pos, s_pos_, num_k_exclude_rope)
        if l_bank.shape[seq_dim] > 0:
            l_mem  = torch.cat([l_bank, p_bank], dim=seq_dim)
            l_pos_ = None if l_pos is None else torch.cat([l_pos, p_pos], dim=seq_dim)
            tgt_l  = self._forward_ca(tgt, l_mem, query_pos, l_pos_, num_k_exclude_rope)
            tgt    = tgt_s + torch.tanh(self.alpha) * tgt_l
        else:
            tgt    = tgt_s
    
        assert p_bank.shape[seq_dim] == num_k_exclude_rope
        assert s_bank.shape[seq_dim] == min(s_bank_len, cond_tokens + noncond_tokens)
        # print("ptr_start=", ptr_start,
        #     " | len(memory)=", total_len,
        #     " | S=", s_bank.shape[seq_dim],
        #     " (cond=", cond_tokens, ", k_nc=", k_nc, ")",
        #     " | L=", l_bank.shape[seq_dim],
        #     " | P=", p_bank.shape[seq_dim],
        #     " | tanh(alpha)=", float(torch.tanh(self.alpha)))

        # 4) Feed-forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)


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
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
        s_bank_len: int = 0,        
        cond_tokens: int = 0,
        noncond_tokens: int = 0,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            kwds["s_bank_len"] = s_bank_len
            kwds["cond_tokens"] = cond_tokens
            kwds["noncond_tokens"] = noncond_tokens
            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
