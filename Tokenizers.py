from transformers import BertTokenizer
from typing import Optional, List

class NaiveTokenizer(BertTokenizer):
    def __init__(
        self, 
        vocab_file, 
        do_lower_case=False,        # already lowercase
        do_basic_tokenize=True,     # split by white space
        never_split=None, 
        unk_token=None,
        sep_token=None,
        pad_token=None, 
        cls_token=None, 
        mask_token=None, 
        **kwargs
    ):
        super().__init__(
            vocab_file, 
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split, 
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token, 
            cls_token=cls_token, 
            mask_token=mask_token, 
            **kwargs
        )
        del self.wordpiece_tokenizer
    
    def _tokenize(self, text):
        split_tokens = self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        return split_tokens

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

        return [self.bos_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.eos_token_id]
        