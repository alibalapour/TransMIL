import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default = [2])
    parser.add_argument('--fold', default = 0)
    
    parser.add_argument('--PLIP_encoder', default=False, action="store_true")
    parser.add_argument('--label_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--log_path', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    
    parser.add_argument('--opt', type=str, default='lookahead_radam')
    parser.add_argument('--lr', type=float, default=0.00004)
    parser.add_argument('--opt_eps', type=float, default=None)
    parser.add_argument('--opt_betas', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight_decay', type=None, default=0.00001)
    
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'PLIP_encoder': args.PLIP_encoder
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        gpus=cfg.General.gpus,
        amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)

    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            
            model.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])
            model.to('cuda')

            trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.Data.label_dir = args.label_dir
    cfg.Data.data_dir = args.data_dir
    cfg.General.log_path = args.log_path
    cfg.Model.checkpoint_path = args.checkpoint_path
    
    cfg.Optimizer.opt = args.opt
    cfg.Optimizer.lr = args.lr
    cfg.Optimizer.opt_eps = args.opt_eps
    cfg.Optimizer.opt_betas = args.opt_betas
    cfg.Optimizer.momentum = args.momentum
    cfg.Optimizer.weight_decay = args.weight_decay

    #---->main
    main(cfg)
 