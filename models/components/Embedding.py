import torch
import torch.nn as nn


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


class Embedding(nn.Module):
    def __init__(self, config, word_embedding=None):
        super().__init__()
        self.identity_map_reordering = config.identity_map_reordering

        if word_embedding is None:
            self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.word_embedding.weight.data.normal_(mean=0.0, std=1.0)
        else:
            self.word_embedding = word_embedding
        
        self.position_embedding = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(config.max_len, config.d_model, 0), freeze=True
        )

        if not self.identity_map_reordering:
            self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, past_key_value=None):
        if past_key_value is not None:
            start_pos = past_key_value[0].shape[2]
        else:
            start_pos = 0

        seq = torch.arange(start_pos, start_pos + input_ids.shape[1], device=input_ids.device).view(1, -1)
        embeddings = self.word_embedding(input_ids) + self.position_embedding(seq)

        if not self.identity_map_reordering:
            embeddings = self.dropout(self.layer_norm(embeddings))
        
        return embeddings
