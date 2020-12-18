# -*-coding=utf-8 -*-

import math
import os
import random
import time
import logging

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch
from torch.cuda import amp
import torch.nn as nn

from models.yolo_fastest import YoloFastest
from dataset.voc_dataset import SimpleDataset
from loss.detection_loss import compute_loss

logger = logging.getLogger(__name__)


def train(params, device):

    # cudnn benchmark
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = False
    cudnn.benchmark = True

    save_path = params["io_params"]["save_path"]
    train_path = params["io_params"]["train_path"]
    input_size = params["io_params"]["input_size"]
    total_epochs = params["train_params"]["total_epochs"]
    batch_size = params["train_params"]["batch_size"]

    model = YoloFastest(params["io_params"]).to(device)
    model.initialize_weights()

    dataset = SimpleDataset(train_path, input_size, augment=True,
                            aug_params=params["augment_params"], rect=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=None,
                            pin_memory=True, collate_fn=SimpleDataset.collate_fn)

    batch_per_epoch = len(dataloader)
    num_warm = max(3*batch_per_epoch, 1e3)

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)

    train_params = params["train_params"]
    optimizer = optim.Adam(model.parameters(), lr=train_params['lr0'], betas=(
        train_params['momentum'], 0.999))

    def lf(x): return (
        ((1 + math.cos(x * math.pi / total_epochs)) / 2) ** 1.0) * 0.8 + 0.2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=True)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        optimizer.zero_grad()

        for batch_id, (imgs, targets) in enumerate(dataloader):
            iteration = batch_id + batch_per_epoch * epoch  # 训练的总迭代次数

            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warm Up
            if iteration <= num_warm:
                xi = [0, num_warm]
                accumulate = max(1, np.interp(iteration, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(
                        iteration, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])

            # Autocast
            with amp.autocast(enabled=True):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(
                    device), params["io_params"]["anchors"])

            scaler.scale(loss).backward()

            logger.info("Total loss: {:.5f}\n Regression loss: {:.5f}, Objectness loss: {:.5f}, Classfication loss: {:.5f}".format(
                loss.item(), loss_items[0].item(), loss_items[1].item(), loss_items[2].item()))

            # Optimize
            if iteration % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        torch.save(model.state_dict(), save_path+"/epoch_"+str(epoch)+'.pt')

    torch.cuda.empty_cache()


if __name__ == '__main__':

    device = torch.device('cuda:0')
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    params = {
        "io_params": {
            "save_path": 'output',
            "train_path": '/home/lance/data/DataSets/quanzhou/coco_style/cyclist',
            "input_size": 640,
            "num_cls":  1,
            "anchors":  [[[30, 61],  [48, 65],  [52, 132]],
                         [[52, 114], [114, 199], [202, 400]]],
        },

        "augment_params": {
            "hsv_h": 0.015,    # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,      # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,      # image HSV-Value augmentation (fraction)
            "degrees": 0.0,    # image rotation (+/- deg)
            "translate": 0.0,  # image translation (+/- fraction)
            "scale": 1.0,      # image scale (+/- gain)
            "shear": 0.0,      # image shear (+/- deg)
            # image perspective (+/- fraction), range 0-0.001
            "perspective": 0.0,
            "flipud": 0.0,     # image flip up-down (probability)
            "fliplr": 0.5,     # image flip left-right (probability)
            "mixup": 0.0,      # image mixup (probability)
        },

        "train_params": {
            "total_epochs": 10,
            "batch_size": 32,
            "lr0": 0.001,         # initial learning rate (SGD=1E-2, Adam=1E-3)
            "momentum": 0.937,   # SGD momentum/Adam beta1
            "weight_decay": 0.0005,
        },

        # "network_params" : {}
    }

    train(params, device)
