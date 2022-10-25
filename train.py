import warnings
from simplet5 import SimpleT5
from opts import parse_args
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def allocate_gpus(gpus: int):
    if gpus > 0:
        import torch
        if torch.cuda.is_available():
            import random, string, os
            import numpy as np

            filename = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > {}'.format(filename))
            gpus_memory = [int(x.split()[2]) for x in open(filename,'r').readlines()]
            os.system('rm {}'.format(filename))

            gpus_list = np.argsort(gpus_memory)[::-1]

            device_string = ','.join(map(str, sorted(gpus_list[:gpus])))
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', device_string)
            print(os.environ['CUDA_VISIBLE_DEVICES'])


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    if args.wise:
        allocate_gpus(args.gpus)

    logger = TensorBoardLogger(args.logger_path)

    trainer = Trainer(
        deterministic=True,
        weights_summary='full',
        max_epochs=args.epochs,
        reload_dataloaders_every_epoch=False,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        logger=logger,
        gpus=args.gpus,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        strategy=DDPPlugin(find_unused_parameters=False),
        resume_from_checkpoint=args.resume_from,
    )

    t5 = SimpleT5()
    t5.prepare(args)
    t5.train(args, trainer)
