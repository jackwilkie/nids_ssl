#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file for SSL Experiments

Created on Tue Jun 13 12:04:00 2023

@author: jack
"""

from self_supervised.tabular.saint import saint, composite_loss

import torch as T
from model_training.training_loops import train
from model_training.dataloaders import unlabelled_tabular_dl
from data_read.limited_data import get_limited_train_set
from data_read.tabular_data import dataset_from_config
from omegaconf import DictConfig, OmegaConf
from loggers import logger
import hydra
import inspect 
from utils import get_id
from utils.optimisers import get_std_opt


def create_model(config_dict, device):
    
    constructor_functions = {
        'saint' : saint.make_default
        }
    
    model_name = config_dict['name']
    args_needed = inspect.signature(constructor_functions[model_name]).parameters.keys()
    args_to_pass = {k: v for k, v in config_dict.items() if k in args_needed}
    args_to_pass['device'] = device
    return constructor_functions[model_name](**args_to_pass)

'''
def train_model(model_config, data_config, hyperparams):
    
    model = create_model(model_config)
    
    x_train, x_test, y_train, y_test = dataset_from_config(data_config)
    
    
    # weights and biases testing
    stats = logger.Logger(project = 'test_run', config = config)
    
    args_needed = inspect.signature(train).parameters.keys()
    args_to_pass = {k: v for k, v in config.items() if k in args_needed}
    
    train(model, **args_to_pass, logger = stats)
    
    stats.finish()


def run_script(script_name, config):

    script_dict = {
        'train': train_model
        }
    
    stats = logger.Logger(project = 'test_run', config = config)
    
    script_name = config['script']
    data_config = config['dataset']
    model_config = config['model']
    hyperparams = config['hyperparameters']
    
    script_dict[script_name](model_config, data_config,hyperparams)
'''    
    
@hydra.main(version_base = None, config_path = 'configs', config_name = 'config')
def main(config: DictConfig):
    
    print('Using config: ')
    print(OmegaConf.to_yaml(config))
    
    
    # create model
    model = create_model(config['model'], config.device)
    
    #import dataset 
    x_train, x_test, y_train, y_test = dataset_from_config(config['dataset'])
    
    x_train, y_train, _, _  = get_limited_train_set(x_train, y_train, benign_samples = 8700, attack_samples = 100)
    
    x_train = T.tensor(x_train, device = config.device)
    x_test = T.tensor(x_test, device = config.device)
    y_train = T.tensor(y_train, device = config.device)

    # get dataloaders
    train_dl = unlabelled_tabular_dl(x_train, batch_size = config['hyperparameters']['batch_size'], device = config['device'])
    val_dl =  unlabelled_tabular_dl(x_test, batch_size = config['hyperparameters']['batch_size'], device = config['device'])
    
    if config.log:
        #stats = logger.Logger(project = config['project'], run_name =  f'{config["run_name"]}_{get_id.get_id()}',config = config)
        stats = logger.Logger(project = config['project'], run_name = None, config = config)
    else:
        stats = None
        
    #optimiser = T.optim.AdamW(model.parameters(), lr= config.hyperparameters.unsupervised_lr, weight_decay= config.hyperparameters.decay)  #define optimiser
    optimiser = get_std_opt(model, config.model.d_model)
    #scheduler = T.optim.lr_scheduler.CosineAnnealingLR(optimiser, config['hyperparameters']['epochs'], verbose=False)
    
    loss = composite_loss.make_composite_loss(
        d_model = config.model.d_model, 
        d_hidden = config.model.d_proj_ff, 
        d_proj = config.model.d_proj, 
        temperature = config.hyperparameters.temperature, 
        n_num = config.model.n_numeric, 
        n_cat = config.model.n_cat, 
        cats = config.model.cats, 
        lambda_pt = config.hyperparameters.lambda_pt,
        device = config.device)
    
    # unsupervised pretraining
    train(model = model,
          optimiser = optimiser,
          loss_calc = loss.calc_loss,
          epochs = config['hyperparameters']['epochs'],
          train_dl = train_dl,
          val_dl = val_dl,
          #scheduler = scheduler,
          logger = stats
          )
         
    
    
if __name__ == '__main__':
    print('starting')
    main()