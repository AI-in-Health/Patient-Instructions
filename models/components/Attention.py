from typing import Optional
import torch
import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_decoder = config.is_decoder

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = getattr(config, 'attention_dropout_rate', config.dropout_rate)
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim)
        self.k = nn.Linear(self.d_model, self.inner_dim)
        self.v = nn.Linear(self.d_model, self.inner_dim)
        self.o = nn.Linear(self.inner_dim, self.d_model)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        output_query_states=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        # real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            # real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        ) # (batch_size, n_heads, seq_length, dim_per_head)
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        ) # (batch_size, n_heads, seq_length, dim_per_head)

        # compute scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2)) / np.sqrt(self.key_value_proj_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)
        
        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        
        if output_query_states:
            return outputs, unshape(query_states)

        return outputs


class AttentionWithGate(Attention):
    def __init__(self, config):
        super().__init__(config)
        self.gate = nn.Linear(self.d_model + self.d_model, self.d_model)
    
    def forward(self, *args, **kwargs):
        attention_output, query_states = super().forward(*args, **kwargs, output_query_states=True)
        hidden_states = attention_output[0] # (batch_size, seq_length, dim)

        alpha = torch.sigmoid(self.gate(torch.cat([query_states, hidden_states], dim=-1))) # (batch_size, seq_length, dim)
        hidden_states = hidden_states * alpha
        
        present_key_value_state = attention_output[1]
        attentions = attention_output[2:] + (alpha, )
        
        return (hidden_states, present_key_value_state, attentions)


class AdditiveAttention(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.w3 = nn.Linear(d_model, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, 
            hidden_states: torch.Tensor, # (bsz, d_model)
            feats: torch.Tensor, # (bsz, seq_len, d_model)
            attention_mask: Optional[torch.Tensor]=None,
    ):
        if len(hidden_states.shape) == 1:
            hidden_states = hidden_states.unsqueeze(0)
        if len(hidden_states.shape) == 3 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
       
        bsz, seq_len, *_ = feats.shape
        d_model = hidden_states.shape[-1]

        hidden_states = hidden_states.unsqueeze(1).repeat(1, seq_len, 1).view(bsz * seq_len, d_model)

        logits = self.w3(self.tanh(self.w1(hidden_states) + self.w2(feats.view(bsz * seq_len, d_model))))
        logits = logits.view(bsz, seq_len)

        if attention_mask is not None:
            logits = torch.masked_fill(logits, ~(attention_mask.bool()), -1e6)

        probs = torch.softmax(logits, dim=1)
        probs = self.dropout(probs)
        context = torch.bmm(probs.unsqueeze(1), feats).squeeze(1) 

        return (context, probs)
