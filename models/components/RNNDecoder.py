import torch
import torch.nn as nn
from models.components.Attention import Attention, AdditiveAttention, AttentionWithGate
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class RNNDecoderBase(nn.Module):
    def __init__(self, config,):
        super().__init__()
        self.config = config
        assert self.config.rnn_type == 'lstm', \
            "We only support rnn_type = `lstm` now! Using `gru` may have some bugs"

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def init_decoder_rnn_hidden_states(self, *args, **kwargs):
        decoder_rnn_hidden_states = None
        return decoder_rnn_hidden_states

    def preparation_before_feedforward(self, decoder_rnn_hidden_states, encoder_hidden_states):
        if decoder_rnn_hidden_states is None:
            decoder_rnn_hidden_states = self.init_decoder_rnn_hidden_states(
                encoder_hidden_states=encoder_hidden_states
            )

        return decoder_rnn_hidden_states, encoder_hidden_states

    def get_mean_video_features(self, encoder_hidden_states):
        if not isinstance(encoder_hidden_states, list):
            encoder_hidden_states = [encoder_hidden_states]

        mean_v = torch.stack(encoder_hidden_states, dim=0).mean(0)
        mean_v = mean_v.mean(1) # [bsz, dim_hidden]
        return mean_v

    def get_hidden_states(self, decoder_rnn_hidden_states):
        if self.config.rnn_type == 'lstm':
            hidden_states = decoder_rnn_hidden_states[0]
        else:
            hidden_states = decoder_rnn_hidden_states
        
        if len(hidden_states.shape) == 3:
            assert hidden_states.size(0) == 1
            hidden_states = hidden_states.squeeze(0)

        return hidden_states

    def forward_step(self, it, encoder_hidden_states, decoder_rnn_hidden_states=None, attention_mask=None, **kwargs):
        raise NotImplementedError('Please implement `forward_step` in the derived classes')
    
    def postprocessing(self, hidden_states, cross_attentions, **kwargs):
        present_key_value_state = None
        return hidden_states, cross_attentions, present_key_value_state
    
    def forward(
        self, 
        input_ids, 
        encoder_hidden_states, 
        return_dict=False, 
        attention_mask=None,
        past_rnn_state=None,
        past_key_value=None,
        use_cache=None,
        **kwargs
    ):
        assert input_ids.dim() == 2, "(bsz, seq_len)"

        all_hidden_states = []
        all_attention_probs = []

        decoder_rnn_hidden_states = past_rnn_state

        for i in range(input_ids.size(1)):
            it = input_ids[:, i]

            outputs = self.forward_step(
                it=it, 
                encoder_hidden_states=encoder_hidden_states,
                decoder_rnn_hidden_states=decoder_rnn_hidden_states,
                attention_mask=attention_mask
            )
            hidden_states, decoder_rnn_hidden_states, *_ = outputs

            all_hidden_states.append(hidden_states)
            if len(_):
                # self.use_mha = False
                # we need to collect the attention weights of the AdditiveAttention
                assert len(_) == 1
                all_attention_probs.append(_[0])
        
        hidden_states = torch.stack(all_hidden_states, dim=1)
        if len(all_attention_probs):
            cross_attentions = torch.stack(all_attention_probs, dim=1)
        else:
            cross_attentions = None
        
        # self.use_mha = True, operate MHA
        hidden_states, cross_attentions, present_key_value_state = self.postprocessing(
            hidden_states=hidden_states, 
            cross_attentions=cross_attentions, 
            encoder_hidden_states=encoder_hidden_states, 
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        if present_key_value_state is not None:
            # self.use_mha = True, this is to speed up the inference process
            # because the key and value states of MHA do not need to re-calculate
            decoder_rnn_hidden_states = decoder_rnn_hidden_states + present_key_value_state

        if not return_dict:
            return (hidden_states, cross_attentions)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=decoder_rnn_hidden_states,
            hidden_states=hidden_states,
            attentions=None,
            cross_attentions=cross_attentions,
        )


class SingleLayerRNNDecoder(RNNDecoderBase):
    def __init__(self, config, word_embedding=None):
        super().__init__(config)
        if word_embedding is None:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.embedding.weight.data.normal_(mean=0.0, std=1.0)
        else:
            self.embedding = word_embedding

        # define the rnn module
        rnn_func = nn.LSTMCell if config.rnn_type == 'lstm' else nn.GRUCell
        self.rnn = rnn_func(
            # inputs: y(t-1)
            input_size=config.d_model,
            hidden_size=config.d_model
        )

        self.use_mha = getattr(config, 'use_mha', False)
        if self.use_mha:
            # dot-product attention is faster and more gpu-efficient
            self.att = Attention(config)
        else:
            self.att = AdditiveAttention(config.d_model, getattr(config, 'attention_dropout_rate', config.dropout_rate))
        
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward_step(
        self, 
        it, 
        encoder_hidden_states, 
        decoder_rnn_hidden_states=None, 
        attention_mask=None,
        **kwargs
    ):
        assert it.dim() == 1, '(bsz, )'

        decoder_rnn_hidden_states, encoder_hidden_states = self.preparation_before_feedforward(
            decoder_rnn_hidden_states, encoder_hidden_states)

        rnn_inputs = [self.embedding(it)]
        rnn_inputs = self.dropout(torch.cat(rnn_inputs, dim=-1))

        # print(type(decoder_rnn_hidden_states))
        # print(len(decoder_rnn_hidden_states))
        # print(rnn_inputs.shape, decoder_rnn_hidden_states)
        decoder_rnn_hidden_states = self.rnn(rnn_inputs, decoder_rnn_hidden_states)
        hidden_states = self.get_hidden_states(decoder_rnn_hidden_states)

        if self.use_mha:
            return hidden_states, decoder_rnn_hidden_states

        context, attention_probs = self.att(
            hidden_states=hidden_states, # use h(t) as the query
            feats=encoder_hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = hidden_states + context
        hidden_states = self.dropout(hidden_states)

        return (hidden_states, decoder_rnn_hidden_states, attention_probs)
    
    def postprocessing(self, hidden_states, cross_attentions, encoder_hidden_states, attention_mask, past_key_value, use_cache, **kwargs):
        if self.use_mha:
            context, present_key_value_state, cross_attentions = self.att(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                key_value_states=encoder_hidden_states,
                past_key_value=past_key_value,
                output_attentions=True,
                use_cache=use_cache,
            )
            hidden_states = hidden_states + context
            hidden_states = self.dropout(hidden_states)
        
            return hidden_states, cross_attentions, present_key_value_state
        else:
            return hidden_states, cross_attentions, tuple()


class SingleLayerRNNDecoderWithPE(SingleLayerRNNDecoder):
    def __init__(self, config, word_embedding=None):
        super().__init__(config, word_embedding=word_embedding)
        assert self.use_mha is True, "Only support using multi-head attention in SingleLayerRNNDecoderWithPE"

        attention_module = AttentionWithGate if config.use_gate else Attention
        if config.n_relevant_info > 0:
            self.RelevantAttention = attention_module(config)
        
        if config.use_knowledge_graph:
            self.GraphAttention = attention_module(config)
    
    def get_kv(self, index, past_key_value):
        if past_key_value is None:
            return index+2, past_key_value
        return index+2, past_key_value[index:index+2]

    def postprocessing(
        self, 
        hidden_states, 
        encoder_hidden_states, 
        attention_mask, 
        past_key_value, 
        use_cache,
        relevant_embs_list,
        graph_embs,
        **kwargs,
    ):
        present_key_value_state = ()
        attentions = ()

        index = 0
        index, past_kv = self.get_kv(index, past_key_value)

        att_input_kwargs = dict(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            output_attentions=True,
            use_cache=use_cache,
            past_key_value=past_kv
        )

        # attend to the encoder outputs (i.e., the report of the patient)
        attention_output = self.att(
            **att_input_kwargs,
            attention_mask=attention_mask,
        )
        
        outputs = hidden_states + attention_output[0]
        present_key_value_state = present_key_value_state + attention_output[1]
        attentions = attentions + attention_output[2:]
        
        if hasattr(self, 'RelevantAttention'):
            # attend to retrieved instructions' embs
            for embs in relevant_embs_list:
                index, past_kv = self.get_kv(index, past_key_value)
                att_input_kwargs['key_value_states'] = embs
                att_input_kwargs['past_key_value'] = past_kv

                relevant_attention_output = self.RelevantAttention(**att_input_kwargs)

                outputs = outputs + relevant_attention_output[0]
                present_key_value_state = present_key_value_state + relevant_attention_output[1]
                attentions = attentions + relevant_attention_output[2:]
        
        if hasattr(self, 'GraphAttention'):
            # attend to knowledge graph's nodes' embs
            if isinstance(graph_embs, torch.Tensor):
                assert graph_embs.dim() == 3
                if graph_embs.shape[0] == 1:
                    graph_embs = graph_embs.repeat(outputs.shape[0], 1, 1) # (batch_size, num_nodes, d_model)

            index, past_kv = self.get_kv(index, past_key_value)
            att_input_kwargs['key_value_states'] = graph_embs
            att_input_kwargs['past_key_value'] = past_kv

            graph_attention_output = self.GraphAttention(**att_input_kwargs)

            outputs = outputs + graph_attention_output[0]
            present_key_value_state = present_key_value_state + graph_attention_output[1]
            attentions = attentions + graph_attention_output[2:]

        outputs = self.dropout(outputs)
        return outputs, attentions, present_key_value_state
