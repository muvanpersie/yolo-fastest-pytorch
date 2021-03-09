# -*-coding=utf-8 -*-

import math
import os
import random
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from models.yolo_fastest import YoloFastest
from dataset.dataset import SimpleDataset
from loss.detection_loss import compute_loss

logger = logging.getLogger(__name__)


def train(params):

    save_path = params["io_params"]["save_path"]
    train_path = params["io_params"]["train_path"]
    input_size = params["io_params"]["input_size"]
    total_epochs = params["train_params"]["total_epochs"]
    batch_size = params["train_params"]["batch_size"]

    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu

    model = YoloFastest(params["io_params"]).cuda()
    model.initialize_weights()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dataset = SimpleDataset(train_path, aug_mode=params["augment_params"]["aug_mode"],
                            aug_params=params["augment_params"])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=None,
                            pin_memory=True, collate_fn=SimpleDataset.collate_fn)

    batch_per_epoch = len(dataloader)
    num_warm = max(3*batch_per_epoch, 1e3)

    nbs = 48  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)

    train_params = params["train_params"]
    optimizer = optim.Adam(model.parameters(), lr=train_params['lr0'], betas=(
        train_params['momentum'], 0.999))

    def lf(epoch): 
        return ((1+math.cos(epoch*math.pi/total_epochs))/2)*0.8+0.2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start_epoch = 0
    scheduler.last_epoch = start_epoch - 1
    
    scaler = amp.GradScaler(enabled=True)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        optimizer.zero_grad()

        for batch_id, (imgs, targets) in enumerate(dataloader):
            iteration = batch_id + batch_per_epoch * epoch  # 训练的总迭代次数

            imgs = imgs.cuda().float() / 255.0

            if iteration <= num_warm:
                xi = [0, num_warm]
                accumulate = max(1, np.interp(iteration, xi, [1, nbs/batch_size]).round())
                for x in optimizer.param_groups:
                    x['lr'] = np.interp(iteration, xi, [0.0, x['initial_lr'] * lf(epoch)])

            # Autocast
            with amp.autocast(enabled=True):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.cuda(), params["io_params"])

            scaler.scale(loss).backward()


            logger.info("Batch: {}-{}, ".format(epoch, batch_id) + "Total loss: {:.4f}\n".format(loss.item()) + 
                        " Reg loss: {:.4f}, Obj loss: {:.4f}, Cls loss: {:.4f}".format(
                           loss_items[0].item(), loss_items[1].item(), loss_items[2].item()))

            # Optimize
            if iteration % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        torch.save(model.module.state_dict() if n_gpu>1 else model.state_dict(), save_path+"/epoch_"+str(epoch)+'.pt')

    torch.cuda.empty_cache()


if __name__ == '__main__':

    # cudnn benchmark
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = False
    cudnn.benchmark = True

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    from config.config import params

    train(params)
