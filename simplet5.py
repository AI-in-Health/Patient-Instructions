from collections import defaultdict
import json, pickle
import os
from typing import Any, Dict, List, Optional
import torch
from tqdm.auto import tqdm
from transformers import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import MeanMetric
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from misc.cocoeval import suppress_stdout_stderr, MIMICScorer
from rouge import Rouge
from dataset import LightningDataModule


class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(self, args, opt_to_override: Dict[str, Any]={}):
        super().__init__()
        self.save_hyperparameters()
        self.args = self.get_args()
        # func `get_tokenzier` should be called first to define args.vocab_size,
        # so that func `get_model` can be called without errors
        self.tokenizer = get_tokenizer(self.args, update_vocab_size=True)
        self.model = get_model(self.args)

        if getattr(args, 'pt_path', None) is not None:
            print('- Loading pre-trained checkpoint from', args.pt_path)
            pretrained_state_dict = torch.load(args.pt_path)
            now_state_dict = self.model.state_dict()
            valid_state_dict = {}

            mapping = {
                'EncDecAttention': 'CrossAttention',
                'DenseReluDense.wi': 'fc1',
                'DenseReluDense.wo': 'fc2',
            }

            incompatible_shapes = []
            unloaded = []

            for k, v in pretrained_state_dict.items():
                for key in mapping:
                    if key in k:
                        k = k.replace(key, mapping[key])
                        break
                        
                if k in now_state_dict:
                    if v.shape == now_state_dict[k].shape:
                        valid_state_dict[k] = v
                    else:
                        incompatible_shapes.append(k + f' (model: {now_state_dict[k].shape}; pt: {v.shape})')
                else:
                    unloaded.append(k)

            print('Incompatible shapes:', incompatible_shapes)
            print('           unloaded:', unloaded)
            print('            missing:', [k for k in now_state_dict if k not in valid_state_dict])

            self.model.load_state_dict(valid_state_dict, strict=False)

        self.prepare_metrics()
    
    def get_args(self):
        args = self.hparams.args
        for k, v in self.hparams.opt_to_override.items():
            setattr(args, k, v)
        return args
    
    def prepare_metrics(self):
        self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None, relevant_embs_list=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            relevant_embs_list=relevant_embs_list
        )

        return output.loss, output.logits

    def feedforward_step(self, batch):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        relevant_embs_list = batch.get('relevant_embs_list', None)

        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
            relevant_embs_list=relevant_embs_list
        )

        batch_size = input_ids.shape[0]
        return loss, logits, batch_size

    def training_step(self, batch, batch_idx):
        """ training step """
        loss, logits, batch_size = self.feedforward_step(batch)
        self.train_loss.update(value=loss, weight=batch_size)
        self.log("global_train_loss", loss, prog_bar=False, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ validation step """
        # loss, logits, batch_size = self.feedforward_step(batch)
        # self.val_loss.update(value=loss, weight=batch_size)
        # self.log("global_val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return self.translate_step(batch)

    def test_step(self, batch, batch_idx):
        """ test step """
        # loss, logits, batch_size = self.feedforward_step(batch)
        # self.test_loss.update(value=loss, weight=batch_size)
        return self.translate_step(batch)

    def configure_optimizers(self):
        """ configure optimizers """
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        if self.args.lr_schedule:
            from torch.optim.lr_scheduler import StepLR
            lr_decay = self.args.lr_decay
            lr_step_size = self.args.lr_step_size

            lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay)
            other_info = {}
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    **other_info
                }
            }
        else:
            return optimizer
    
    def configure_callbacks(self):
        # if args.save_topk_models > 1:
        #     some_args_about_checkpoint = {
        #         'save_top_k': args.save_topk_models,
        #         'filename': 'E{epoch:02d}-loss{val_loss:.3f}',
        #         'auto_insert_metric_name': False,
        #     }
        # else:
        some_args_about_checkpoint = {
            'save_top_k': 1,
            'filename': 'best',
            # 'save_top_k': 50,
            # 'filename': "best-{val_Bleu_4:.3f}-{epoch:02d}",
        }

        self.checkpoint_callback = ModelCheckpoint(
            monitor=self.args.monitor_metric,
            mode=self.args.monitor_mode,
            save_last=True,
            dirpath=self.args.checkpoint_path,
            **some_args_about_checkpoint
        )
        
        all_callbacks = [LearningRateMonitor(logging_interval='step'), self.checkpoint_callback]
        
        if getattr(self.args, 'early_stop', False):
            early_stop_callback = EarlyStopping(
                monitor='best_' + self.args.monitor_metric, # best validation score (averaged over all gpus!!!)
                mode=self.args.monitor_mode,
                min_delta=0.00, 
                patience=self.args.patience, 
                verbose=False, 
            )
            all_callbacks.append(early_stop_callback)
        
        return all_callbacks

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.log('train_loss', self.train_loss.compute().item(), prog_bar=True, sync_dist=True, batch_size=self.train_loss.weight.item())
        self.train_loss.reset()
    
    def validation_epoch_end(
        self, 
        validation_step_outputs, 
        best_to_be_recorded=['val_METEOR', 'val_R-1', 'val_R-2', 'val_R-L', 'val_Bleu_1', 'val_Bleu_2', 'val_Bleu_3', 'val_Bleu_4']
    ) -> None:
        # self.log('val_loss', self.val_loss.compute().item(), prog_bar=True, sync_dist=True, batch_size=self.val_loss.weight.item())
        # self.val_loss.reset()

        scores, n_samples = self.evaluation(
            validation_step_outputs, return_n_samples=True, prefix='val_')
        
        if not hasattr(self, 'best_monitor_metric') or scores[self.args.monitor_metric] > self.best_monitor_metric:
            self.best_monitor_metric = scores[self.args.monitor_metric]
            self.best_scores = scores

        scores['best_{}'.format(self.args.monitor_metric)] = self.best_monitor_metric

        best_to_be_recorded = [metric for metric in best_to_be_recorded if metric != self.args.monitor_metric]
        for metric in best_to_be_recorded:
            if not hasattr(self, metric) or scores[metric] > getattr(self, metric):
                setattr(self, metric, scores[metric])
            
            scores['best_{}'.format(metric)] = getattr(self, metric)

        for k, v in scores.items():
            self.log(k, v, prog_bar=True if k == 'best_{}'.format(self.args.monitor_metric) else False, sync_dist=True, batch_size=n_samples)
        
        tqdm.write(f'======================= Epoch {self.current_epoch} =======================')
        for metric in ['Bleu_4', 'METEOR', 'R-L']:
            now = scores['val_'+metric]
            best = self.best_scores['val_'+metric]
            tqdm.write('{}\tnow: {:.6f}\tbest: {:.6f}\tdiff: {}{:.6f}'.format(
                metric, now, best, '+' if now > best else '-', now - best 
            ))
        self.trainer._results

    
    def test_epoch_end(self, test_step_outputs) -> None:
        # self.log('test_loss', self.test_loss.compute().item(), sync_dist=True, batch_size=self.test_loss.weight.item())
        # self.test_loss.reset()

        scores, n_samples = self.evaluation(
            test_step_outputs, return_n_samples=True, prefix='test_')

        for k, v in scores.items():
            self.log(k, v, sync_dist=True, batch_size=n_samples)
    

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None) # don't show the version number
        return items

    def translate_step(self, batch):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        relevant_embs_list = batch.get('relevant_embs_list', None)
        graph_embs = batch.get('graph_embs', None)

        generated_ids = self.model.generate(
            input_ids=input_ids, 
            max_length=self.args.max_len,
            attention_mask=attention_mask,
            num_beams=self.args.beam_size,
            length_penalty=self.args.beam_alpha,
            num_return_sequences=1,
            repetition_penalty=self.args.repetition_penalty,
            relevant_embs_list=relevant_embs_list,
            graph_embs=graph_embs,
        )

        if getattr(self.args, 'save_attentions', False):
            self.save_attentions(batch, generated_ids)

        preds = defaultdict(list)
        assert len(generated_ids) == len(batch['key'])

        DEBUG = getattr(self.args, 'DEBUG', False)
        for k, ids in zip(batch['key'], generated_ids):
            pred = self.tokenizer.decode(ids, skip_special_tokens=True)
            preds[k].append({'image_id': k, 'caption': pred})
            if DEBUG:
                print(k, pred)

        # print(preds)
        return preds
    
    def evaluation(self, 
            all_step_outputs: Dict[str, List[dict]], 
            scorer: MIMICScorer = MIMICScorer(),
            references: Optional[Dict[str, List[dict]]] = None,
            return_n_samples: bool = False,
            prefix: str = '',
            save_csv: bool = False,
            csv_path: Optional[str] = None, 
            csv_name: Optional[str] = None,
        ):
        preds = {}
        for item in all_step_outputs:
            preds.update(item)
        
        if references is None:
            references = self.get_coco_style_references()
        
        with suppress_stdout_stderr():
            scores, detailed_scores = scorer.score(GT=references, RES=preds, IDs=preds.keys())
            scores['Bleu'] = (scores['Bleu_1'] + scores['Bleu_2'] + scores['Bleu_3'] + scores['Bleu_4']) / 4
            scores.update(self.get_rouge_scores(preds, references))

            if getattr(self.args, 'save_json', False):
                self.save_json(preds, detailed_scores)
        
        scores['Sum'] = sum([scores[metric] for metric in self.args.sum_metrics])

        if getattr(self.args, 'save_csv', False) or save_csv:
            total_len = 0
            for k in preds:
                this_len = len(preds[k][0]['caption'].split(' '))
                total_len += this_len
            scores['avg_length'] = total_len * 1.0 / len(preds)
            self.save_scores_to_csv(scores, csv_path, csv_name)
        
        if prefix:
            scores = {prefix+k: v for k, v in scores.items()}

        if return_n_samples:
            return scores, len(preds)

        return scores
    
    def get_rouge_scores(self, hyps, gts, scorer=Rouge()):
        HYPS, GTS = [], []
        
        for k in hyps.keys():
            if not hyps[k][0]['caption'].strip():
                # if hyp is a empty string, `get_scores` will raise a error
                continue

            HYPS.append(hyps[k][0]['caption'])
            GTS.append(gts[k][0]['caption'])
        
        if len(HYPS):
            scores = scorer.get_scores(HYPS, GTS, avg=True)
            return {
                'R-1': scores['rouge-1']['f'] * len(HYPS) / len(hyps), 
                'R-2': scores['rouge-2']['f'] * len(HYPS) / len(hyps), 
                'R-L': scores['rouge-l']['f'] * len(HYPS) / len(hyps)
            }
        return {'R-1': 0, 'R-2': 0, 'R-L': 0}

    def get_coco_style_references(self) -> Dict[str, List[dict]]:
        if not hasattr(self, 'references'):
            val_df = load_csv(self.args.val_csv_path, extract_columns=['key', 'target_text'])
            test_df = load_csv(self.args.test_csv_path, extract_columns=['key', 'target_text'])
            
            self.references = defaultdict(list)
            for df in [val_df, test_df]:
                for i in range(len(df)):
                    data = df.iloc[i]
                    self.references[data['key']].append({'image_id': data['key'], 'caption': data['target_text']})

        return self.references

    def save_scores_to_csv(self, scores, csv_path=None, csv_name=None):
        if csv_path is None:
            csv_path = self.args.csv_path
        if csv_name is None:
            csv_name = self.args.csv_name

        os.makedirs(csv_path, exist_ok=True)
        csv_file = os.path.join(csv_path, csv_name)
        
        if not os.path.exists(csv_file):
            f = open(csv_file, 'w')
            f.write(','.join(self.args.csv_filednames) + '\n')
        else:
            f = open(csv_file, 'a')
        
        csv_keys = [getattr(self.args, key) for key in self.args.csv_keys]
        scores['key'] = self.args.csv_key_format.format(*csv_keys)
        scores['epoch'] = self.current_epoch
        scores['repetition_penalty'] = self.args.repetition_penalty
        scores['beam_size'] = self.args.beam_size

        data = [str(scores[k]) for k in self.args.csv_filednames]
        f.write(','.join(data) + '\n')
        f.close()

        for k in ['key', 'epoch', 'repetition_penalty', 'beam_size']:
            scores.pop(k)
    
    def save_json(self, preds, detailed_scores):
        assert hasattr(self.args, 'save_base_path') and self.args.save_base_path
        assert hasattr(self.args, 'save_folder') and self.args.save_folder
        assert hasattr(self.args, 'json_file_name') and self.args.json_file_name

        save_path = os.path.join(self.args.save_base_path, self.args.save_folder)
        os.makedirs(save_path, exist_ok=True)
        
        assert len(detailed_scores.keys()) == len(preds.keys())

        for k in detailed_scores:
            assert len(preds[k]) == 1
            detailed_scores[k]['instruction'] = preds[k][0]['caption']

        json.dump(
            detailed_scores, 
            open(os.path.join(save_path, self.args.json_file_name), 'w')
        )
    
    def save_attentions(self, batch, generated_ids):
        assert hasattr(self.args, 'save_base_path') and self.args.save_base_path
        assert hasattr(self.args, 'save_folder') and self.args.save_folder
        assert hasattr(self.args, 'attentions_folder_name') and self.args.attentions_folder_name

        save_path = os.path.join(self.args.save_base_path, self.args.save_folder, self.args.attentions_folder_name)
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["source_text_input_ids"],
                attention_mask=batch["source_text_attention_mask"],
                relevant_embs_list=batch.get('relevant_embs_list', None),
                labels=generated_ids,
                decoder_attention_mask=generated_ids.ne(0),
                return_dict=True,
                output_attentions=True,
            )
        
        def tensor2numpy(tensor, index=0):
            if tensor is None:
                return None

            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy()[index]

            new_tensor = ()
            for item in tensor:
                if isinstance(item, tuple):
                    if len(item) == 1:
                        new_item = item[0]
                    else:
                        new_item = item
                else:
                    new_item = item
                
                if isinstance(new_item, torch.Tensor):
                    new_item = new_item.cpu().numpy()[index]
                else:
                    assert isinstance(new_item, tuple)
                    assert isinstance(new_item[0], torch.Tensor)
                    new_item = tuple([i.cpu().numpy()[index] for i in new_item])
                
                new_tensor = new_tensor + (new_item, )

            return new_tensor

        for batch_idx, key in enumerate(batch['key']):
            data = dict(
                encoder_attentions=tensor2numpy(outputs['encoder_attentions'], index=batch_idx),
                decoder_attentions=tensor2numpy(outputs.get('decoder_attentions', None), index=batch_idx),
                cross_attentions=tensor2numpy(outputs['cross_attentions'], index=batch_idx),
            )
            this_save_path = os.path.join(save_path, f'{key}.pkl')
            pickle.dump(data, open(this_save_path, 'wb'))


def load_csv(
        path, 
        columns={"discharge_instruction": "target_text", "discharge_summary": "source_text"}, 
        extract_columns=['key', 'source_text', 'target_text'],
    ):
    import pandas as pd
    df = pd.read_csv(path)
    df = df.rename(columns=columns)
    df = df[extract_columns]
    return df


def get_tokenizer(args, update_vocab_size=True):
    from Tokenizers import NaiveTokenizer
    import os
    assert os.path.exists(args.vocab_path)
    tokenizer = NaiveTokenizer.from_pretrained(args.vocab_path)
    if update_vocab_size:
        args.vocab_size = tokenizer.vocab_size
    return tokenizer


def get_model(args):
    import os
    import json
    from config.Config import DSConfig

    assert os.path.exists(args.config_path), args.config_path

    config_kwargs = json.load(open(args.config_path, 'rb'))
    config_kwargs['vocab_size'] = args.vocab_size

    if args.dropout_rate is not None:
        config_kwargs['dropout_rate'] = args.dropout_rate

    if getattr(args, 'num_layers', None) is not None:
        config_kwargs['num_layers'] = args.num_layers

    if args.num_decoder_layers is not None:
        config_kwargs['num_decoder_layers'] = args.num_decoder_layers

    if args.model in ['vanilla', 'lstm']:
        if args.model == 'vanilla':
            from models.vanilla_transformer import Transformer
            config = DSConfig(**config_kwargs)
            model = Transformer(config)
        elif args.model == 'lstm':
            from models.rnn import Seq2Seq
            config_kwargs['rnn_type'] = 'lstm'
            config = DSConfig(**config_kwargs)
            model = Seq2Seq(config)

    elif args.model in ['variant', 'lstm_pe']:
        if args.not_use_retrieval:
            config_kwargs['n_relevant_info'] = 0
        else:
            print('relevant_concat', getattr(args, 'relevant_concat', False))
            if getattr(args, 'relevant_concat', False):
                config_kwargs['n_relevant_info'] = 1
            else:
                config_kwargs['n_relevant_info'] = len(args.relevant_info_paths)

        config_kwargs['d_embs'] = 768 * int(args.embs_path.split('_')[-2])
        
        config_kwargs['embedder_ln'] = args.embedder_ln

        config_kwargs['use_knowledge_graph'] = args.use_knowledge_graph #TODO
        config_kwargs['use_gate'] = args.use_gate
        config_kwargs['sqrt_scale'] = args.sqrt_scale
        config_kwargs['relevant_topk'] = args.relevant_topk
        config_kwargs['rank_embs'] = getattr(args, 'rank_embs', False)

        if args.use_knowledge_graph:
            config_kwargs['adjacent_matrix_path'] = args.adjacent_matrix_path
            config_kwargs['adjacent_matrix_counts_path'] = getattr(args, 'adjacent_matrix_counts_path', None)
            config_kwargs['gcn_freq'] = getattr(args, 'gcn_freq', False)
            config_kwargs['adjacent_matrix_threshold'] = args.adjacent_matrix_threshold
            config_kwargs['gcn_num_layers'] = args.gcn_num_layers
            config_kwargs['gcn_bert_embs_path'] = args.gcn_bert_embs_path
            config_kwargs['normalize_method'] = args.normalize_method

        if args.model == 'variant':
            from models.variant_transformer import TransformerWithPE
            config = DSConfig(**config_kwargs)
            model = TransformerWithPE(config)
        elif args.model == 'lstm_pe':
            from models.rnn import Seq2SeqWithPE
            config_kwargs['rnn_type'] = 'lstm'
            config = DSConfig(**config_kwargs)
            model = Seq2SeqWithPE(config)
    else:
        raise ValueError('args.model should be in [`vanilla`, `variant`, `lstm`, `lstm_pe`]')
    
    print(model)
    return model


def get_data_module(args, tokenizer, mode=None):
    return LightningDataModule(args, tokenizer, mode)


class SimpleT5:
    """ Custom SimpleT5 class """

    def __init__(self) -> None:
        """ initiates SimpleT5 class """

    def prepare(self, args):
        self.wrapper = LightningModel(args)
        self.data_module = get_data_module(args, tokenizer=self.wrapper.tokenizer)
    
    def train(self, args, trainer: Optional[pl.Trainer] = None):
        pl.seed_everything(args.seed)
        trainer.fit(self.wrapper, self.data_module)
        
        print('best_model_path:', self.wrapper.checkpoint_callback.best_model_path)
        print('best_model_score', self.wrapper.checkpoint_callback.best_model_score)

        self.wrapper = LightningModel.load_from_checkpoint(self.wrapper.checkpoint_callback.best_model_path)
        trainer.test(self.wrapper, self.data_module)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model

        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.

        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds

    def load_pretrained_model(self, model_path, opt_to_override = {}):
        self.wrapper = LightningModel.load_from_checkpoint(model_path, strict=True, opt_to_override=opt_to_override)
    
    def load_data_module(self, args=None, tokenizer=None):
        if args is None:
            args = self.wrapper.get_args()
        if tokenizer is None:
            tokenizer = self.wrapper.tokenizer
        
        if getattr(args, 'translate_csv_fn', None) is not None:
            ori_path = getattr(args, f'{args.mode}_csv_path')
            new_path = os.path.join(os.path.dirname(ori_path), args.translate_csv_fn)
            setattr(args, f'{args.mode}_csv_path', new_path)
            print(f'- Loading `{args.mode}` data from {new_path}')
        
        if getattr(args, 'relevant_postfix', None) is not None:
            new_paths = []
            for p in args.relevant_info_paths:
                root, fn = os.path.dirname(p), os.path.basename(p)
                new_paths.append(os.path.join(root, fn.split('.')[0] + args.relevant_postfix + '.pkl'))
                
            args.relevant_info_paths = new_paths
            print(f'- Loaindg relevant info from', args.relevant_info_paths)

        self.data_module = get_data_module(args, tokenizer=tokenizer, mode=getattr(args, 'mode', None))

    def load(self, model_path: str, opt_to_override: Dict[str, Any] = {}) -> None:
        self.load_pretrained_model(model_path, opt_to_override)
        self.load_data_module()        
    
    def evaluate(self, trainer, mode='test'):
        func = getattr(trainer, mode)
        func(self.wrapper, self.data_module)
