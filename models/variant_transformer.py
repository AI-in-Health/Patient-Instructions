""" Variant Transformer Model. """
from models import vanilla_transformer as vanilla
import torch
import torch.nn as nn
from models.components.Attention import AttentionWithGate, Attention


class Embedder(nn.Module):
    def __init__(self, d_input, d_model, dropout_rate, layer_norm=False, layer_norm_epsilon=1e-6, rank_embs=False, topk=None):
        super().__init__()
        if rank_embs:
            assert topk is not None
            self.rank_embs = nn.Embedding(topk, d_input)
            self.rank_embs.weight.data.normal_(mean=0.0, std=1.0)

        self.net = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        if layer_norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.constant_(self.net[0].bias, 0)
    
    def forward(self, inputs):
        if isinstance(inputs, list):
            return [self.forward(i) for i in inputs]
        
        if hasattr(self, 'rank_embs'):
            inputs = inputs + self.rank_embs.weight[None, :, :]
        
        output = self.net(inputs)

        if hasattr(self, 'layer_norm'):
            output = self.layer_norm(output)

        return output


class LayerCrossAttentionWithPE(vanilla.LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)

        attention_module = AttentionWithGate if config.use_gate else Attention
        if config.n_relevant_info > 0:
            self.RelevantAttention = attention_module(config)
        
        if config.use_knowledge_graph:
            self.GraphAttention = attention_module(config)
    
    def merge(self, att_input_kwargs, hidden_states, attention_output, relevant_embs_list, graph_embs, other_past_key_value):
        merged_output = attention_output[0]

        present_key_value_state = attention_output[1]
        attentions = attention_output[2:]

        att_input_kwargs['attention_mask'] = None
        att_input_kwargs['past_key_value'] = None
        index = 0

        if hasattr(self, 'RelevantAttention'):
            for embs in relevant_embs_list:
                att_input_kwargs['key_value_states'] = embs

                if other_past_key_value is not None:
                    att_input_kwargs['past_key_value'] = other_past_key_value[index:index+2]
                    index += 2

                relevant_attention_output = self.RelevantAttention(**att_input_kwargs)

                merged_output = merged_output + relevant_attention_output[0]
                present_key_value_state = present_key_value_state + relevant_attention_output[1]
                attentions = attentions + relevant_attention_output[2:]
        
        if hasattr(self, 'GraphAttention'):
            if isinstance(graph_embs, torch.Tensor):
                assert graph_embs.dim() == 3
                if graph_embs.shape[0] == 1:
                    graph_embs = graph_embs.repeat(merged_output.shape[0], 1, 1) # (batch_size, num_nodes, d_model)

            att_input_kwargs['key_value_states'] = graph_embs

            if other_past_key_value is not None:
                att_input_kwargs['past_key_value'] = other_past_key_value[index:index+2]
                index += 2

            graph_attention_output = self.GraphAttention(**att_input_kwargs)

            merged_output = merged_output + graph_attention_output[0]
            present_key_value_state = present_key_value_state + graph_attention_output[1]
            attentions = attentions + graph_attention_output[2:]

        hidden_states = hidden_states + self.dropout(merged_output)
        return hidden_states, (present_key_value_state, attentions)


class TransformerWithPE(vanilla.Transformer):
    def __init__(self, config):
        vanilla.LayerCrossAttention = LayerCrossAttentionWithPE
        super().__init__(config)

        if config.n_relevant_info > 0:
            self.relevant_net = Embedder(
                d_input=config.d_embs,
                d_model=config.d_model,
                dropout_rate=config.dropout_rate,
                layer_norm=config.embedder_ln,
                layer_norm_epsilon=config.layer_norm_epsilon,
                rank_embs=config.rank_embs,
                topk=config.relevant_topk,
            )
        
        if config.use_knowledge_graph:
            from models.components.GCN import GraphNet
            self.graph_net = GraphNet(config)

    def project_relevant_embs(self, relevent_embs_list):
        if hasattr(self, 'relevant_net'):
            assert relevent_embs_list is not None
            return self.relevant_net(relevent_embs_list)
        
        return relevent_embs_list
    
    def project_graph_embs(self, graph_embs):
        return graph_embs

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        data = super().prepare_inputs_for_generation(input_ids, **kwargs)
        data['relevant_embs_list'] = kwargs['relevant_embs_list']
        data['graph_embs'] = kwargs['graph_embs']
        return data
    
    def forward(self, *args, **kwargs):
        graph_embs = 0 # here we do not set it to None because of func: project in Attention
        if hasattr(self, 'graph_net') and kwargs.get('past_key_values', None) is None:
            if not self.training and hasattr(self, 'graph_embs'):
                graph_embs = self.graph_embs
            else:
                graph_embs = self.graph_net()
                graph_embs = graph_embs.unsqueeze(0)
                if not self.training:
                    self.graph_embs = graph_embs
                elif hasattr(self, 'graph_embs'):
                    del self.graph_embs

        kwargs['graph_embs'] = graph_embs
        return super().forward(*args, **kwargs)
        
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs = None,
        **model_kwargs,
    ):
        bsz = input_ids.shape[0]
        expanded_return_idx = (
            torch.arange(bsz).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        
        if model_kwargs["relevant_embs_list"] is not None:
            model_kwargs["relevant_embs_list"] = [item.index_select(0, expanded_return_idx) for item in model_kwargs["relevant_embs_list"]]

        return input_ids, model_kwargs

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AttentionWithGate):
            nn.init.xavier_uniform_(module.gate.weight)
            nn.init.constant_(module.gate.bias, 0)
