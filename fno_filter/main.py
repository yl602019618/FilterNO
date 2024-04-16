from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl
import yaml
import argparse 
from bisect import bisect
import os
import torch
import shutil
import warnings
import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#wandb.init(project="filter")
wandb.init(mode = 'disabled')
os.environ['NCCL_P2P_DISABLE']='1'
from FNO import FNO2d
from dataloader import datasetFactory


def main(config_file):
    torch.set_float32_matmul_precision("medium")
    
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    
    c_nn = config['model']
    c_train = config['train'] 
    print(c_nn)
    
    model = FNO2d(modes1 = c_nn['mode'], 
                 modes2 = c_nn['mode'], 
                 width = c_nn['width'], 
                 padding = c_nn['padding'], 
                 input_dim = c_nn['in'], 
                 output_dim = c_nn['out'],
                 loss = c_nn['loss'],
                    learning_rate = c_train['lr'], 
                    step_size= c_train['step_size'],
                    gamma= c_train['gamma'],
                    weight_decay= c_train['weight_decay'],
                    eta_min = c_train['eta_min'])
    print(model)
    c_proj = config['Project']
    print(c_proj)
    if c_proj['checkpoint'] == False:
        save_file = os.path.join(config["ckpt"]["PATH"], 
                                config["ckpt"]["save_dir"])
        checkpoint_callback = ModelCheckpoint(                                
                                dirpath=save_file,
                                every_n_epochs = 1,
                                save_last = True,
                                monitor = 'val_loss',
                                mode = 'min',
                                save_top_k = c_proj['save_top_k'],
                                filename="model-{epoch:03d}-{val_loss:.4f}",
                            )

    if os.path.exists(save_file):
        print(f"The model directory exists. Overwrite? {c_proj['erase']}")
        if c_proj['erase'] == True:
            shutil.rmtree(save_file)



    train_dataloader, val_dataloader = datasetFactory(config=config, do=c_proj['do'])
    max_epochs = config["train"]["epochs"] 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    if c_proj['devices'] == 1 :
        trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=c_proj['accelerator'], 
                            devices = c_proj['devices'],
                            callbacks = [checkpoint_callback,lr_monitor])#,
                            #strategy = 'deepspeed',gradient_clip_val=0.8)  # dp ddp deepspeed
    else: 
        device_num = [i for i in range(c_proj['devices'])]
        trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=c_proj['accelerator'], 
                            devices = device_num,
                            callbacks = [checkpoint_callback,lr_monitor],
                            strategy = 'deepspeed',gradient_clip_val=0.8)
    trainer.fit(model, train_dataloader, val_dataloader)
    if c_proj['save'] == True:
        save_path = os.path.join(c_proj['save_path'], 'model.pt')
        torch.save(model, save_path)
        print('save model done')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Training of the Architectures', add_help=True)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='/home/bcl/zhijunzeng/NBSO_official_new/config/NBSO.yaml')
    args=parser.parse_args()
    config_file = args.config_file
    main(config_file)