# -*-coding=utf-8 -*-

import argparse
import math
import os
import random
import time
import logging

import numpy as np
# import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader 
import yaml
import torch
from torch.cuda import amp

from models.yolo import YoloFastest
from dataset.voc_dataset import SimpleDataset
# from utils.general import (
#     labels_to_class_weights, check_anchors)

from loss.detection_loss import compute_loss
# from utils.torch_utils import init_seeds, ModelEMA

logger = logging.getLogger(__name__)



def train(hyp, opt, device):

    # cudnn benchmark
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = False
    cudnn.benchmark = True
    
    log_dir = 'output/'
    root_path = '/home/lance/data/DataSets/quanzhou/coco_style/cyclist/images'
    num_cls = 1
    names = ['cyclist']
    imgsz = 640

    epochs, batch_size = opt.epochs, opt.batch_size
    
    model = YoloFastest().to(device)
    # ema = ModelEMA(model)  # 指数滑动平均
    
    dataset = SimpleDataset(root_path, imgsz, batch_size, augment=True, hyp=hyp, stride=32)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=None,
                            pin_memory=True, collate_fn=SimpleDataset.collate_fn)
    
    batch_per_epoch = len(dataloader)
    num_warm = max(3*batch_per_epoch, 1e3)

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    
    # Model parameters
    hyp['cls'] = num_cls  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = num_cls  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, num_cls).to(device)
    model.names = names

    # Check anchors
    # if not opt.noautoanchor:
    #     check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
    

    t0 = time.time()
    
    start_epoch = 0
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=True)

    for epoch in range(start_epoch, epochs): 
        model.train()
        optimizer.zero_grad()

        mloss = torch.zeros(4, device=device)  # mean losses for each epoch
        for batch_id, (imgs, targets, _, _) in enumerate(dataloader):
            num_iter = batch_id + batch_per_epoch * epoch  # 训练的总迭代次数
            
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warmup
            if num_iter <= num_warm:
                xi = [0, num_warm]

                accumulate = max(1, np.interp(num_iter, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, [0.9, hyp['momentum']])

            # Autocast
            with amp.autocast(enabled=True):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
            
            scaler.scale(loss).backward()

            # Optimize
            if num_iter % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                # if ema:
                #     ema.update(model)

                mloss = (mloss * batch_id + loss_items) / (batch_id + 1)  # update mean losses
            
            print("loss: ", loss.item(),  loss_items)

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # if ema:
        #     ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == epochs
        
        # Save model
        ckpt = {'epoch': epoch,
                'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                'optimizer': None if final_epoch else optimizer.state_dict()}
        torch.save(ckpt, log_dir+"/epoch_"+str(epoch)+'.pt')
        del ckpt

    logger.info('%g epochs completed in %.3f minutes.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 60))

    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training') # 默认为False
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--logdir', type=str, default='output/', help='logging directory')
    opt = parser.parse_args()
   
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)

    device = torch.device('cuda:0')
    
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger.info(opt)

    
    hyp_path = 'data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml'
    with open(hyp_path) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    hyp = {
        "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
        "degrees": 0.0,  # image rotation (+/- deg)
        "translate": 0.1,  # image translation (+/- fraction)
        "scale": 0.5,  # image scale (+/- gain)
        "shear": 0.0,  # image shear (+/- deg)
        "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
        "flipud": 0.0,  # image flip up-down (probability)
        "fliplr": 0.5,  # image flip left-right (probability)
        "mixup": 0.0, # image mixup (probability)
    }

    # Train
    train(hyp, opt, device)
