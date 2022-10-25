import argparse
import os
from config import Constants

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-nl', '--num_layers', type=int)
    parser.add_argument('-ndl', '--num_decoder_layers', type=int)

    parser.add_argument('-model', '--model', type=str, default='lstm', choices=['vanilla', 'variant', 'lstm', 'lstm_pe'], 
        help=   'vanilla: Transformer; '
                'variant: Transformer with Prior Experience as addition input; '
                'lstm: LSTM; '
                'lstm_pe: LSTM with Prior Experience as addition input; '
                'This augment specifies the type of decoder while keeps the encoder unchanged (Transformer encoder)')
    parser.add_argument('-arch', '--arch', type=str, choices=['small', 'base'])
    parser.add_argument('-setup', '--setup', type=str, default='naive')
    parser.add_argument('-scope', '--scope', type=str, default='')
    parser.add_argument('-wise', '--wise', default=False, action='store_true', 
            help='if True, 1) allocate gpus with maximun memory for training '
            'and 2) call translate.py with the `--save_csv` argument after training')

    parser.add_argument('-base_data_path', '--base_data_path', type=str)
    parser.add_argument('-base_checkpoint_path', '--base_checkpoint_path', type=str)
    parser.add_argument('--logger_path', type=str, default='./logs')
    parser.add_argument('-csv_mid_path', '--csv_mid_path', type=str, default='splits')
    parser.add_argument('-csv_names', '--csv_names', type=str, nargs='+', default=['train.csv', 'val.csv', 'test.csv'])
    parser.add_argument('-vocab_path', '--vocab_path', type=str, default='./data/vocab')
    parser.add_argument('-config_path', '--config_path', type=str, default='./config/archs/small.json')

    parser.add_argument('-dr', '--dropout_rate', type=float)

    parser.add_argument('-upe', '--use_prior_experience', default=False, action='store_true')
    parser.add_argument('-ug', '--use_gate', default=False, action='store_true')
    parser.add_argument('--relevant_info_paths', type=str, nargs='+', default=['./data/info/DxRelevantInfo.pkl', './data/info/MedRelevantInfo.pkl', './data/info/PrRelevantInfo.pkl'])
    parser.add_argument('--embs_path', type=str, default='./data/instruction_embs/bert-base-uncased_1_max.pkl')
    parser.add_argument('--relevant_topk', type=int, default=20)
    parser.add_argument('--rank_embs', default=False, action='store_true')
    parser.add_argument('-rc', '--relevant_concat', default=False, action='store_true')

    parser.add_argument('-nur', '--not_use_retrieval', default=False, action='store_true')
    parser.add_argument('--embedder_ln', default=False, action='store_true')
    parser.add_argument('--sqrt_scale', default=False, action='store_true')

    # use_knowledge_graph && GCN
    parser.add_argument('-ukg', '--use_knowledge_graph', default=False, action='store_true')
    parser.add_argument('--adjacent_matrix_path', type=str, default='./data/info/adjacent_matrix.npy')
    parser.add_argument('--adjacent_matrix_counts_path', type=str, default='./data/info/diag_counts.npy')
    parser.add_argument('-gcn_freq', '--gcn_freq', default=False, action='store_true')
    parser.add_argument('--adjacent_matrix_threshold', type=int, default=10)
    parser.add_argument('--gcn_num_layers', type=int, default=1)
    parser.add_argument('--gcn_bert_embs_path', type=str)
    parser.add_argument('--normalize_method', type=str, default='freq')

    parser.add_argument('-src_len', '--source_max_token_len', type=int, default=2048)
    parser.add_argument('-trg_len', '--target_max_token_len', type=int, default=512)
    parser.add_argument('-max_len', '--max_len', type=int, default=512)
    parser.add_argument('-rp', '--repetition_penalty', type=float, default=2.5)

    train = parser.add_argument_group(title='Training Settings')
    train.add_argument('-rf', '--resume_from', type=str)

    train.add_argument('-seed', '--seed', type=int, default=0)
    train.add_argument('-bsz', '--batch_size', type=int, default=8)
    train.add_argument('-e', '--epochs', type=int, default=50)
    
    train.add_argument('--lr_schedule', default=False, action='store_true')
    train.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    train.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    train.add_argument('--lr_decay', default=0.9, type=float, help='the decay rate of learning rate per epoch')
    train.add_argument('--lr_step_size', default=1, type=int, help='period of learning rate decay')

    train.add_argument('-gpus', '--gpus', type=int, default=1)
    train.add_argument('-num_workers', '--num_workers', type=int, default=4)

    train.add_argument('--check_val_every_n_epoch', type=int, default=1)
    train.add_argument('--save_topk_models', type=int, default=1)
    train.add_argument('--monitor_metric', type=str, default='val_Bleu_4')
    train.add_argument('--monitor_mode', type=str, default='max')
    train.add_argument('--sum_metrics', type=str, nargs='+', default=['R-1', 'R-2', 'R-L'])
    train.add_argument('--num_sanity_val_steps', type=int, default=0)
    train.add_argument('-agb', '--accumulate_grad_batches', type=int)
    train.add_argument('--limit_train_batches', type=float, default=1.0)
    train.add_argument('--limit_val_batches', type=float, default=1.0)
    train.add_argument('--limit_test_batches', type=float, default=1.0)

    train.add_argument('-early', '--early_stop', default=False, action='store_true')
    train.add_argument('-patience', '--patience', type=int, default=10)

    train.add_argument('--pt_path', type=str)

    eval = parser.add_argument_group(title='Evaluation Settings')
    eval.add_argument('--beam_size', type=int, default=2)
    eval.add_argument('--beam_alpha', type=float, default=1.0)

    args = parser.parse_args()

    load_yaml(args, args.setup, './config/setups.yaml')

    if hasattr(args, 'scope_format'):
        format_str, args_list = args.scope_format

        all_str = []
        for a in args_list:
            assert hasattr(args, a)
            this_str = str(getattr(args, a))
            if '.' in this_str:
                this_str = this_str.split('.')[0]
            all_str.append(this_str)
        
        scope = format_str.format(*all_str)
        if args.scope:
            args.scope = '_'.join([scope, args.scope])
        else:
            args.scope = scope
        
    assert args.scope, '`scope` should not be empty, please pass the `--scope XXX` argument; the model will be saved to $base_checkpoint_path/$scope'
    
    args.logger_path = os.path.join(args.logger_path, args.scope)
    if args.arch is not None:
        args.config_path = os.path.join('./config/archs', args.arch + '.json')

    prepare_dataset(args)
    prepare_checkpoint_path(args)

    print(f'- setup: {args.setup}')
    print(f'- epochs: {args.epochs}')
    print(f'- batch_size: {args.batch_size}')
    print(f'- early stopping: {args.early_stop}')
    print(f'- patience: {args.patience}')
    print(f'- scope: {args.scope}')
    
    return args

def prepare_dataset(args):
    if args.base_data_path is None:
        args.base_data_path = Constants.base_data_path
    
    assert len(args.csv_names) == 3, '- You should specify csv names for `train`, `val` and `test` sets.'
    for k, name in zip(['train', 'val', 'test'], args.csv_names):
        key = f'{k}_csv_path'
        setattr(args, key, os.path.join(args.base_data_path, args.csv_mid_path, name))
    
    del args.csv_names
    del args.csv_mid_path
    
def prepare_checkpoint_path(args):
    if args.base_checkpoint_path is None:
        args.base_checkpoint_path = Constants.base_checkpoint_path

    args.checkpoint_path = os.path.join(
        args.base_checkpoint_path,
        args.scope
    )

import yaml
def load_yaml(args, key, yaml_path=None, yaml_data=None, modify_scope=False):
    if not key or key is None:
        return None

    assert yaml_path is not None or yaml_data is not None
    if yaml_data is None:
        yaml_data = yaml.full_load(open(yaml_path))
    
    assert key in yaml_data.keys(), f"`{key}` can not be found in {yaml_path}"

    specific_data = yaml_data[key]

    if 'inherit_from' in specific_data.keys():
        inherit_from = specific_data.pop('inherit_from')
        if isinstance(inherit_from, list):
            for new_key in inherit_from:
                load_yaml(args, key=new_key, yaml_path=yaml_path, yaml_data=yaml_data)    
        else:
            load_yaml(args, key=inherit_from, yaml_path=yaml_path, yaml_data=yaml_data)

    new_scope = key
    format_str = None
    if modify_scope:
        if 'scope_format' in specific_data:
            format_str, names = specific_data.pop('scope_format')
        elif hasattr(args, 'scope_format'):
            format_str, names = args.scope_format
            del args.scope_format

    for k, v in specific_data.items():
        setattr(args, k, v)
        
    if modify_scope:
        if format_str is not None:
            new_scope = format_str.format(*[getattr(args, name) for name in names])
        args.scope = new_scope + '_' + args.scope if args.scope else new_scope
