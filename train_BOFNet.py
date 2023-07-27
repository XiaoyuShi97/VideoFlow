from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer
import core.datasets_3frames as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger

from core.Networks import build_network

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg):

    loss_func = sequence_loss
    if cfg.use_smoothl1:
        print("[Using smooth L1 loss]")
        loss_func = sequence_loss_smooth

    model = nn.DataParallel(build_network(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            images, flows, valids = [x.cuda() for x in data_blob]
            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                images = (images + stdv * torch.randn(*images.shape).cuda()).clamp(0.0, 255.0)

            output = {}
            flow_predictions = model(images, output)
            loss, metrics, _ = loss_func(flow_predictions, flows, valids, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                # torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'sintel_train':
                        results.update(evaluate_tile.validate_sintel(model.module))

                logger.write_dict(results)
                
                model.train()
            
            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--validation', type=str, nargs='+')

    args = parser.parse_args()

    if args.stage == 'things':
        from configs.things import get_cfg    
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    train(cfg)
