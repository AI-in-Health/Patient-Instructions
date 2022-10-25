""" Vanilla Seq2Seq Model. """
import copy
import torch
import torch.nn as nn
from models import vanilla_transformer as vanilla
from models.components.RNNDecoder import SingleLayerRNNDecoder, SingleLayerRNNDecoderWithPE
from models.components.Attention import AdditiveAttention, AttentionWithGate
from transformers.modeling_outputs import Seq2SeqLMOutput


class Seq2Seq(vanilla.VanillaPreTrainedModel):
    def __init__(self, config, RNN_decoder_module=SingleLayerRNNDecoder):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.max_len = config.encoder_max_len

        self.encoder = vanilla.Stack(encoder_config, self.shared)
        
        self.use_mha = True
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_mha = self.use_mha
        self.decoder = RNN_decoder_module(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AdditiveAttention):
            nn.init.xavier_uniform_(module.w1.weight)
            nn.init.xavier_uniform_(module.w2.weight)
            nn.init.xavier_uniform_(module.w3.weight)
            nn.init.constant_(module.w1.bias, 0)
            nn.init.constant_(module.w2.bias, 0)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def project_relevant_embs(self, relevent_embs_list):
        return relevent_embs_list
    
    def project_graph_embs(self, graph_embs):
        return graph_embs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
        relevant_embs_list=None,
        graph_embs=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        encoder_hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            
            boundary = 2 if self.config.rnn_type == 'lstm' else 1
            past_rnn_state = past_key_values[:boundary] 
            past_key_value = past_key_values[boundary:]
        else:
            past_rnn_state, past_key_value = None, None
        
        relevant_embs_list = self.project_relevant_embs(relevant_embs_list)
        graph_embs = self.project_graph_embs(graph_embs)        

        if self.use_mha:
            attention_mask = self.invert_attention_mask(attention_mask)
        else:
            attention_mask = attention_mask

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            relevant_embs_list=relevant_embs_list,
            graph_embs=graph_embs,
            return_dict=return_dict,
            attention_mask=attention_mask,
            past_rnn_state=past_rnn_state,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.config.d_model ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
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

        assert encoder_outputs is not None
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )
        model_kwargs["encoder_outputs"] = encoder_outputs
        
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        assert past is not None
        activate_past = tuple([p.index_select(0, beam_idx.to(p.device)) for p in past])
        return activate_past


class Seq2SeqWithPE(Seq2Seq):
    def __init__(self, config, RNN_decoder_module=SingleLayerRNNDecoderWithPE):
        super().__init__(config, RNN_decoder_module=RNN_decoder_module)

        if config.n_relevant_info > 0:
            from models.variant_transformer import Embedder
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
    
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AttentionWithGate):
            if isinstance(module.gate, nn.Sequential):
                nn.init.xavier_uniform_(module.gate[0].weight)
                nn.init.constant_(module.gate[0].bias, 0)
                nn.init.xavier_uniform_(module.gate[2].weight)
            else:
                nn.init.xavier_uniform_(module.gate.weight)
                nn.init.constant_(module.gate.bias, 0)
    
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
    
    def project_relevant_embs(self, relevent_embs_list):
        if hasattr(self, 'relevant_net'):
            assert relevent_embs_list is not None
            return self.relevant_net(relevent_embs_list)
        
        return relevent_embs_list

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        data = super().prepare_inputs_for_generation(input_ids, **kwargs)
        data['relevant_embs_list'] = kwargs['relevant_embs_list']
        data['graph_embs'] = kwargs['graph_embs']
        return data
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
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

        assert encoder_outputs is not None
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )
        model_kwargs["encoder_outputs"] = encoder_outputs
        
        if model_kwargs["relevant_embs_list"] is not None:
            model_kwargs["relevant_embs_list"] = [item.index_select(0, expanded_return_idx) for item in model_kwargs["relevant_embs_list"]]
        
        if model_kwargs["graph_embs"] is not None:
            pass

        return input_ids, model_kwargs
