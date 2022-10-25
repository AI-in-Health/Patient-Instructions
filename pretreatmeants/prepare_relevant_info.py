import argparse
import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
import torch
from typing import List, Dict
from collections import defaultdict

def get_admission_ids(splits_path='./data/splits', splits_names=['train', 'val', 'test']):
    assert os.path.exists(splits_path)
    admission_ids = []
    keys = []
    n_train_admission_ids = None
    assert len(splits_names) == 3
    assert 'train' in splits_names[0]

    for name in splits_names:
        csv_path = os.path.join(splits_path, name+'.csv')
        assert os.path.exists(csv_path)

        data = pd.read_csv(csv_path)
        for i in range(len(data)):
            key = data.iloc[i]['key']
            keys.append(key)
            admission_id = int(key.split('_')[1])
            admission_ids.append(admission_id)
        
        if 'train' in name:
            n_train_admission_ids = len(admission_ids)
            assert n_train_admission_ids == len(data)

    return admission_ids, n_train_admission_ids, keys


def mapping_age_to_range(age, range=[[0, 55], [55, 70], [70, 200]]):
    for begin, end in range:
        if begin <= age < end:
            return f'{begin}_{end}'


def get_code_map(pickle_path: str, train_admission_ids: List[int]):
    assert os.path.exists(pickle_path)
    data = pickle.load(open(pickle_path, 'rb'))
    code_map = {}
    index = 0
    for admission_id in train_admission_ids:
        if admission_id not in data:
            continue
        for item in data[admission_id]:
            if item not in code_map:
                code_map[item] = index
                index += 1
    print(pickle_path, len(code_map))
    return code_map


def prepare_code_embs(pickle_path: str, embs_path: str, admission_ids: List[int], code_map: Dict[str, int]):
    if os.path.exists(embs_path):
        print(f'- load embs from {embs_path}')
        return torch.from_numpy(np.load(embs_path))

    assert os.path.exists(pickle_path)
    
    embs = np.zeros((len(admission_ids), len(code_map)))
    data = pickle.load(open(pickle_path, 'rb'))
    for i, admission_id in enumerate(admission_ids):
        if admission_id not in data:
            continue
        
        this_emb = np.zeros((len(code_map,)))
        for item in data[admission_id]:
            if item not in code_map:
                # val or test samples
                continue
            this_emb[code_map[item]] = 1
        
        embs[i, :] = this_emb
    
    np.save(embs_path, embs)
    print(f'- Have saved embs to {embs_path}')
    return torch.from_numpy(embs)


def get_topk_relevant_given_embs(
        embs: torch.Tensor, 
        n_train_admission_ids: int, 
        keys: List[str], 
        topk: int = 300, 
        tmp_fn: str='tmp',
    ):
    train_embs = embs[:n_train_admission_ids, :]

    relevant_info = defaultdict(list)
    if os.path.exists(tmp_fn):
        with open(tmp_fn, 'r') as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                this_id, relevant_ids = line.split('\t')
                relevant_ids = relevant_ids.split(',')
                relevant_info[this_id] = relevant_ids
        
        f = open(tmp_fn, 'a')
    else:
        f = open(tmp_fn, 'w')
    
    for i, (emb, key) in tqdm(enumerate(zip(embs, keys))):
        if key in relevant_info:
            continue

        sum = emb.sum()
        if sum == 0:
            # this admission does not have any codes
            continue
        
        scores = torch.cosine_similarity(emb.unsqueeze(0), train_embs).view(-1)

        if i < n_train_admission_ids:
            scores[i] = 0 # exclude itself
        
        _, topk_indices = scores.topk(topk, dim=0, largest=True, sorted=True)
        
        for j in topk_indices:
            relevant_info[keys[i]].append(keys[j])

        f.write('{}\t{}\n'.format(keys[i], ','.join([str(item) for item in relevant_info[keys[i]]])))

    f.close()
    os.remove(tmp_fn)
    
    return relevant_info


def run_topk_relevant(
        args,
        topk=300, 
        splits_path='./splits', 
        splits_names=['train', 'val', 'test'],
        base_path='./diagnose-procedure-medication',
        alias_list=['Dx', 'Med', 'Pr'],
        embs_base_path=None,
        info_base_path=None
    ):
    assert os.path.exists(base_path), base_path

    admission_ids, n_train_admission_ids, keys = get_admission_ids(splits_path, splits_names)
    
    code_map_list = []
    for alias in alias_list:
        pickle_path = os.path.join(base_path, f'adm{alias}Map_mimic3.pk')
        code_map = get_code_map(pickle_path, train_admission_ids=admission_ids[:n_train_admission_ids])
        code_map_list.append(code_map)
        print('there are {} unique codes for {}'.format(len(code_map), alias))
    
    assert len(code_map_list) == len(alias_list)
    print('-' * 100)

    for code_map, alias in zip(code_map_list, alias_list):
        pickle_path = os.path.join(base_path, f'adm{alias}Map_mimic3.pk')
        embs_path = os.path.join(base_path if embs_base_path is None else embs_base_path, f'{alias}Embs.npy')
        embs = prepare_code_embs(pickle_path, embs_path, admission_ids, code_map)

        info_path = os.path.join(base_path if info_base_path is None else info_base_path, f'{alias}RelevantInfo.pkl')
        if not os.path.exists(info_path):
            if torch.cuda.is_available():
                embs = embs.cuda()

            relevant_info = get_topk_relevant_given_embs(embs, n_train_admission_ids, keys, topk, tmp_fn=alias)
            pickle.dump(relevant_info, open(info_path, 'wb'))
        else:
            relevant_info = pickle.load(open(info_path, 'rb'))

        print('there are {} - {} = {} admission ids that have empty relevant info'.format(
            len(embs), len(relevant_info), len(embs) - len(relevant_info)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_save_path', type=str, default='./data')
    parser.add_argument('--alias_list', type=str, nargs='+', default=['Dx', 'Med', 'Pr'])
    args = parser.parse_args()

    embs_base_path = os.path.join(args.base_save_path, 'embs')
    info_base_path = os.path.join(args.base_save_path, 'info')

    os.makedirs(embs_base_path, exist_ok=True)
    os.makedirs(info_base_path, exist_ok=True)
    run_topk_relevant(
        args,
        topk=50, 
        base_path='./data/diagnose-procedure-medication', 
        splits_path='./data/splits', 
        embs_base_path=embs_base_path, 
        info_base_path=info_base_path, 
        alias_list=args.alias_list,
    )
