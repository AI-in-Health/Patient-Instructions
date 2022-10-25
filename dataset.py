from typing import List
import torch
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
from Tokenizers import NaiveTokenizer
import os


class TextOnlyDataset(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        pad_to_the_longest: bool = False
    ):
        """
        initiates a PyTorch Dataset Module for input data

        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

        if pad_to_the_longest:
            self.kwargs = dict(padding='longest')
        else:
            self.kwargs = dict(
                padding='max_length',
                truncation=True,
            )
        

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        
        key = data_row["key"]

        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            return_attention_mask=True,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.source_max_token_len,
            **self.kwargs
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.target_max_token_len,
            **self.kwargs
        )

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
        ] = -100  # to make sure we have correct labels for T5 text generation

        return dict(
            source_text=source_text,
            target_text=data_row["target_text"],
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
            key=key,
        )


class EmbsLoader(object):
    def __init__(
        self,
        relevant_info_paths: List[str],
        embs_path: str,
        topk: int,
        dim: int,
        concat: float=False,
    ):
        self.relevant_info = [pickle.load(open(path, 'rb')) for path in relevant_info_paths]
        self.embs = pickle.load(open(embs_path, 'rb'))
        self.topk = topk
        self.dim = dim
        self.concat = concat

    def get_embs_based_on_the_key(self, key):
        relevant_embs_list = []
        for info in self.relevant_info:
            this_embs = np.zeros((self.topk, self.dim))
            if key in info:
                relevant_list = info[key]
                relevant_list = relevant_list[:self.topk]

                for i, relevant_key in enumerate(relevant_list):
                    this_embs[i, :] = self.embs[relevant_key]
            
            relevant_embs_list.append(torch.FloatTensor(this_embs))
        
        if self.concat:
            relevant_embs_list = [torch.cat(relevant_embs_list, dim=0)] # [(topk * n_info, dim)]
        
        return dict(relevant_embs_list=relevant_embs_list)


class Dataset(TextOnlyDataset):
    def __init__(self, args, tokenizer, type: str = 'joint', mode: str = 'train'):
        assert type in ['joint', 'text'], type
        assert mode in ['train', 'val', 'test']
        self.type = type
        self.mode = mode

        data = load_csv(getattr(args, f'{mode}_csv_path'))
        if hasattr(args, 'subtask_file'):
            print('- Loading the subtask file from', args.subtask_file)
            keys = set(open(args.subtask_file, 'r').read().strip().split('\n'))
            frames = []
            for i in range(len(data)):
                if data.iloc[i]['key'] in keys:
                    frames.append(data.iloc[i:i+1])
            data = pd.concat(frames, ignore_index=True)
        print(mode, data.shape)

        TextOnlyDataset.__init__(
            self, 
            data=data,
            tokenizer=tokenizer,
            source_max_token_len=args.source_max_token_len,
            target_max_token_len=args.target_max_token_len,
            pad_to_the_longest=getattr(args, 'pad_to_the_longest', False)
        )

        if self.type == 'joint':
            self.embs_loader = EmbsLoader(
                relevant_info_paths=args.relevant_info_paths,
                embs_path=args.embs_path,
                topk=args.relevant_topk,
                dim=768 * int(args.embs_path.split('_')[-2]),
                concat=getattr(args, 'relevant_concat', False),
            )
    
    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        
        if self.type == 'joint':
            item.update(self.embs_loader.get_embs_based_on_the_key(item['key']))
        
        return item


class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(self, args, tokenizer, mode=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers
        self.mode = mode

    def setup(self, stage=None):
        this_type = 'joint' if getattr(self.args, 'use_prior_experience', False) else 'text'
        if self.mode is None or self.mode == 'train':
            self.train_dataset = Dataset(self.args, self.tokenizer, mode='train', type=this_type)
        if self.mode is None or self.mode == 'val':
            self.val_dataset = Dataset(self.args, self.tokenizer, mode='val', type=this_type)
        if self.mode is None or self.mode == 'test':
            self.test_dataset = Dataset(self.args, self.tokenizer, mode='test', type=this_type)

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True, 
            num_workers=self.args.num_workers, persistent_workers=True
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, shuffle=False, 
            num_workers=self.args.num_workers, persistent_workers=True
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, shuffle=False, 
            num_workers=self.args.num_workers, persistent_workers=True
        )


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
