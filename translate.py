import os
import glob
import argparse
import warnings
from pytorch_lightning import Trainer
from simplet5 import SimpleT5
from pytorch_lightning.plugins import DDPPlugin
from config import Constants

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('model_path', type=str)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--DEBUG', default=False, action='store_true')

    save = parser.add_argument_group(title='Save Sth Settings')
    save.add_argument('--save_base_path', type=str, default='./inference_results')
    save.add_argument('--save_folder', type=str, default='')
    save.add_argument('--save_json', default=False, action='store_true')
    save.add_argument('--json_file_name', type=str, default='preds_and_scores.json')

    save.add_argument('--save_attentions', default=False, action='store_true')
    save.add_argument('--attentions_folder_name', type=str, default='attentions')

    cs = parser.add_argument_group(title='Common Settings')
    cs.add_argument('-bsz', '--batch_size', type=int)
    cs.add_argument('-gpus', '--gpus', type=int, default=1)
    cs.add_argument('-num_workers', '--num_workers', type=int, default=8)
    cs.add_argument('-bs', '--beam_size', type=int, default=2)
    cs.add_argument('-ba', '--beam_alpha', type=float, default=1.0)
    cs.add_argument('-lvb', '--limit_val_batches', type=float, default=1.0)
    cs.add_argument('-ltb', '--limit_test_batches', type=float, default=1.0)
    cs.add_argument('-max_len', '--max_len', type=int, default=512)
    cs.add_argument('-rp', '--repetition_penalty', type=float, default=2.5)

    parser.add_argument('--save_csv', default=False, action='store_true')
    parser.add_argument('--csv_path', type=str, default='./csv_results')
    parser.add_argument('--csv_name', type=str, default='overall.csv')
    parser.add_argument('--csv_filednames', type=str, nargs='+', default=['key', 'METEOR', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'Bleu', 'R-1', 'R-2', 'R-L', 'Sum', 'avg_length'])
    parser.add_argument('--csv_key_format', type=str, default='{}_bs{:d}_rp{:.1f}')
    parser.add_argument('--csv_keys', type=str, nargs='+', default=['scope', 'beam_size', 'repetition_penalty'])

    parser.add_argument('--pad_to_the_longest', default=False, action='store_true')

    parser.add_argument('--subtask_type', type=str, help='the script will perform the specified subtask (e.g., for different ages)', choices=['age', 'disease', 'sex'])
    parser.add_argument('--translate_csv_fn', type=str)
    parser.add_argument('--relevant_postfix', type=str)
    parser.add_argument('--relevant_topk', type=int)
    parser.add_argument('--scope', type=str)
    args = parser.parse_args()

    if args.save_csv:
        assert args.gpus == 1
    if args.batch_size is None:
        del args.batch_size
    if args.scope is None:
        del args.scope
    if args.relevant_topk is None:
        del args.relevant_topk

    trainer = Trainer(
        logger=None,
        gpus=args.gpus,
        strategy=DDPPlugin(find_unused_parameters=False),
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )
    
    t5 = SimpleT5()
    if args.subtask_type is not None:
        args.save_csv = True
        args.gpus = 1
        args.csv_keys = ['setup', 'subtask_type', 'subtask_key']
        args.csv_key_format = '{}-{}-{}'

        t5.load_pretrained_model(args.model_path, opt_to_override=vars(args))

        subtask_path = os.path.join(Constants.base_data_path, 'splits', 'subtasks', args.subtask_type, args.mode)
        assert os.path.exists(subtask_path), "We can not find the subtask path %s" % subtask_path

        for f in glob.glob(os.path.join(subtask_path, '*.txt')):
            t5.wrapper.args.subtask_file = f
            t5.wrapper.args.subtask_key = os.path.basename(f).split('.')[0]
            t5.load_data_module()
            t5.evaluate(trainer, mode=args.mode)
    else:
        t5.load(args.model_path, opt_to_override=vars(args))
        t5.evaluate(trainer, mode=args.mode)
