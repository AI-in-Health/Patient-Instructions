import argparse
import json
import os
import torch
from transformers import BertTokenizer, AutoModel
import pandas as pd
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='translate.py')
parser.add_argument('-arch', '--arch', type=str, default='bert-base-uncased')
parser.add_argument('-nl', '--n_layers', type=int, default=1)
parser.add_argument('-pt', '--pooling_type', type=str, default='max')
parser.add_argument('-cp', '--config_path', type=str, default='./data/vocab/tokenizer_config.json')
parser.add_argument('-dp', '--data_path', type=str, default='./data/splits')
parser.add_argument('-s', '--splits', nargs='+', type=str, default=['train', 'val', 'test'])
parser.add_argument('-sp', '--save_path', type=str, default='./data/instruction_embs')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.arch)
tokenizer.basic_tokenizer.never_split = set(json.load(open(args.config_path, 'rb'))['never_split'])
model = AutoModel.from_pretrained(args.arch)
model.cuda()

results = {}
for split in tqdm(args.splits):
    csv_path = os.path.join(args.data_path, split+'.csv')
    data = pd.read_csv(csv_path)

    for i in tqdm(range(len(data))):
        key = data.iloc[i]['key']
        text = data.iloc[i]['discharge_instruction']

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
        for k in inputs.keys():
            inputs[k] = inputs[k].cuda()
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs['hidden_states'][::-1]

        tmp = []
        for j in range(args.n_layers):
            if args.pooling_type == 'max':
                tmp.append(hidden_states[j][0, 1:-1, :].max(dim=0)[0])
            else:
                tmp.append(hidden_states[j][0, 1:-1, :].mean(dim=0))
        
        results[key] = torch.cat(tmp, dim=0).cpu().numpy()

os.makedirs(args.save_path, exist_ok=True)
save_fn = f'{args.arch}_{args.n_layers}_{args.pooling_type}.pkl'
save_path = os.path.join(args.save_path, save_fn)

pickle.dump(results, open(save_path, 'wb'))
