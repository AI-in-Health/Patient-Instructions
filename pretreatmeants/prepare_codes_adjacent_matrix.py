import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
from typing import List
from tqdm import tqdm
import argparse


def get_admission_ids(splits_path='./splits', splits_names=['train', 'val', 'test']):
    assert os.path.exists(splits_path)
    admission_ids = []
    n_train_admission_ids = None
    assert len(splits_names) == 3
    assert 'train' in splits_names[0]

    for name in splits_names:
        csv_path = os.path.join(splits_path, name+'.csv')
        assert os.path.exists(csv_path)

        data = pd.read_csv(csv_path)
        for i in range(len(data)):
            admission_id = int(data.iloc[i]['key'].split('_')[1])
            admission_ids.append(admission_id)
        
        if 'train' in name:
            n_train_admission_ids = len(admission_ids)
            assert n_train_admission_ids == len(data)

    return admission_ids, n_train_admission_ids


def get_code_map(pickle_path: str, train_admission_ids: List[int], index=None):
    assert os.path.exists(pickle_path)
    data = pickle.load(open(pickle_path, 'rb'))
    code_map = {}
    if index is None:
        index = 0
    for admission_id in train_admission_ids:
        if admission_id not in data:
            continue
        for item in data[admission_id]:
            if item not in code_map:
                code_map[item] = index
                index += 1
    return code_map, index, data


def run(code_map, all_data, train_admission_ids):
    adjacent_matrix = np.zeros((len(code_map), len(code_map)))
    counts = np.zeros((len(code_map), 1))

    for admission_id in tqdm(train_admission_ids):
        this_item = set()
        for data in all_data:
            if admission_id in data:
                this_item = this_item | set(data[admission_id])
        
        this_item = list(this_item)
        for item in this_item:
            _id = code_map[item]
            counts[_id] += 1

        length = len(this_item)
        for i in range(length-1):
            for j in range(i+1, length):
                i_id = code_map[this_item[i]]
                j_id = code_map[this_item[j]]
                adjacent_matrix[i_id, j_id] += 1
                adjacent_matrix[j_id, i_id] += 1

    return adjacent_matrix, counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alias_list', type=str, nargs='+', default=['Dx', 'Med', 'Pr'])
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--base_data_path', type=str, default='./data')
    
    parser.add_argument('--base_save_path', type=str, default='./data/info')
    parser.add_argument('--matrix_path', type=str, default='adjacent_matrix.npy')
    parser.add_argument('--code_map_path', type=str, default='adjacent_matrix_code_map.pkl')
    parser.add_argument('--counts_path', type=str, default='diag_counts.npy')
    args = parser.parse_args()

    for k in ['matrix_path', 'code_map_path', 'counts_path']:
        setattr(args, k, os.path.join(args.base_save_path, getattr(args, k)))

    os.makedirs(args.base_save_path, exist_ok=True)

    admission_ids, n_train_admission_ids = get_admission_ids(os.path.join(args.base_data_path, 'splits'))

    index = None
    code_map = {}
    all_data = []
    for code in ['Dx', 'Med', 'Pr']:
        pickle_path = os.path.join(args.base_data_path, 'diagnose-procedure-medication/adm{}Map_mimic3.pk'.format(code))
        this_map, index, data = get_code_map(pickle_path, admission_ids[:n_train_admission_ids], index=index)
        
        code_map.update(this_map)
        all_data.append(data)
    
    adjacent_matrix, counts = run(code_map, all_data, admission_ids[:n_train_admission_ids])

    np.save(args.matrix_path, adjacent_matrix)
    np.save(args.counts_path, counts)
    pickle.dump(code_map, open(args.code_map_path, 'wb'))
