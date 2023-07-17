#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file for SSL Experiments

Created on Tue Jun 13 12:04:00 2023

@author: jack
"""

from self_supervised.tabular.saint import saint, composite_loss

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model_training.training_loops import train
from model_training.dataloaders import tabular_dl
from data_read.limited_data import get_limited_train_set
from data_read.tabular_data import dataset_from_config
from omegaconf import DictConfig, OmegaConf
from loggers import logger
import hydra
import inspect 
from utils.optimisers import get_std_opt
from self_supervised.probes import LayeredProbe, KNNProbe, TripletProbe, ArcProbe
import copy
from metric_learning.losses.triplet_losses import Batch_All_Triplet_Loss
from utils.gpu import get_gpu_num

def create_model(name: str, config: dict = None):
    
    constructor_functions = dict(
        saint = saint.make_default
        )
    
    args_needed = inspect.signature(constructor_functions[name]).parameters.keys()
    args_to_pass = {k: v for k, v in config.items() if k in args_needed}
    return constructor_functions[name](**args_to_pass)


def get_probe(name: str, config: dict):
    
    args = copy.deepcopy(config)
    
    probes = dict(
        linear = LayeredProbe,
        knn = KNNProbe,
        triplet = TripletProbe,
        arcface = ArcProbe,
        mlp = LayeredProbe
        )
    
    activations = dict(
        relu = nn.ReLU
        )
    
    losses = dict(
        cross_entropy = F.cross_entropy
        )        
    
    if 'm' in args: losses['triplet'] = Batch_All_Triplet_Loss(m = config['m'], device = config['device'])
    if args['loss']: args['loss'] = losses[config['loss']]
    if args['activation']: args['activation'] = activations[config['activation']]
        
    return probes[name](**args)


@hydra.main(version_base = None, config_path = 'configs', config_name = 'config')
def main(config: DictConfig):
    
    print('Using config: ')
    print(OmegaConf.to_yaml(config))
    
    device = get_gpu_num()
    
    # create model
    model_config = OmegaConf.to_container(config.model, resolve=True)  # Convert to dict
    model_config['device'] = device  # add model device from run config
    model_config['n_numeric'] = config.dataset.n_numeric # add number of features from dataset config
    model_config['n_cat'] = config.dataset.n_cat
    model_config['cats'] = config.dataset.cats
    model = create_model(model_config['name'], model_config)
    model.to(device)
    
    #import dataset 
    x_train, x_test, y_train, y_test = dataset_from_config(config['dataset'])
    x_train, y_train, _, _  = get_limited_train_set(x_train, y_train, benign_samples = 8700, attack_samples = 100)
    
    x_train = T.tensor(x_train, device = device)
    x_test = T.tensor(x_test, device = device)
    y_train = T.tensor(y_train, device = device)
    y_test = T.tensor(y_test, device = device)
    
    # get dataloaders
    train_dl = tabular_dl(x_train, y = y_train, batch_size = config['hyperparameters']['batch_size'], balanced = False, device = device)
    val_dl =  tabular_dl(x_test, y= y_test, batch_size = config['hyperparameters']['batch_size'], balanced = False, device = device)
    
    #create logger
    if config.log:
        #stats = logger.Logger(project = config['project'], run_name =  f'{config["run_name"]}_{get_id.get_id()}',config = config)
        stats = logger.Logger(project = config['project'], run_name = None, config = config)
    else:
        stats = None
        
        
    optimiser = get_std_opt(model, config.model.d_model, warmup = config.hyperparameters.optimiser_warmup)
    
    
    loss = composite_loss.make_composite_loss(
        d_model = config.model.d_model, 
        d_hidden_contrastive = config.model.d_hidden_contrastive,
        d_proj_contrastive = config.model.d_proj_contrastive,
        d_hidden_reconstructive = config.model.d_hidden_reconstructive,
        d_proj_reconstructive = config.model.d_proj_reconstructive,
        temperature = config.hyperparameters.temperature, 
        n_num = config.dataset.n_numeric, 
        n_cat = config.dataset.n_cat, 
        cats = config.dataset.cats, 
        lambda_pt = config.hyperparameters.lambda_pt,
        contrastive_reduction = config.model.contrastive_reduction,
        device = device)
    
    train_probe_config = OmegaConf.to_container(config.training_probe, resolve=True)  # Convert to dict
    if 'd_out' not in train_probe_config: train_probe_config['d_out'] = config.dataset.n_classes
    train_probe_config['d_model'] = config.model.d_model
    train_probe_config['n_features'] = config.dataset.n_numeric + config.dataset.n_cat + 1
    train_probe_config['n_classes'] = config.dataset.n_classes
    train_probe_config['device'] = device
    train_probe = get_probe(train_probe_config['name'], train_probe_config)    
    train_probe.to(device)
    
    
    # unsupervised pretraining
    train(model = model,
          optimiser = optimiser,
          loss_calc = loss.calc_loss,
          epochs = config.hyperparameters.epochs,
          train_dl = train_dl,
          val_dl = val_dl,
          logger = stats,
          eval_func = train_probe.train_eval,
          eval_interval = config.eval_interval,
          ep_log_interval= 0
          )
    
    
    # get dataloaders
    if 'batch_size' in config.eval_probe:
        bs = config.eval_probe.batch_size
    else:
        bs = config.hyperparameters.batch_size
    
    train_dl = tabular_dl(x_train, y = y_train, batch_size = bs, balanced = True, device = device)
    val_dl =  tabular_dl(x_test, y= y_test, batch_size = bs, balanced = True, device = device)
    
    eval_probe_config = OmegaConf.to_container(config.eval_probe, resolve=True)  # Convert to dict
    if 'd_out' not in eval_probe_config: eval_probe_config['d_out'] = config.dataset.n_classes
    eval_probe_config['d_model'] = config.model.d_model
    eval_probe_config['n_features'] = config.dataset.n_numeric + config.dataset.n_cat + 1
    eval_probe_config['n_classes'] = config.dataset.n_classes
    eval_probe_config['device'] = device
    eval_probe = get_probe(eval_probe_config['name'], eval_probe_config)
    eval_probe.to(device)
    
    eval_probe.to(device)
    metrics = eval_probe.train_eval(model, train_dl, val_dl)
    print(metrics)
    print('training complete')
    
    
if __name__ == '__main__':
    main()