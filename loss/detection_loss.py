# -*- coding=utf-8 -*-

import glob
import math
import os
import random
import shutil
import subprocess
import time
import logging
from contextlib import contextmanager
from copy import copy
from pathlib import Path
import platform

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps



def compute_loss(pred, targets, anchors):

    num_anchor = len(anchors[0])

    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
   
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])).to(device)

    cp, cn = smooth_BCE(eps=0.0)
    
    tcls, tbox, indices, anchors = build_targets(pred, targets, anchors)

    n_scale = len(pred)
    balance = [4.0, 1.0] if n_scale == 2 else [4.0, 1.0, 0.4]
    
    for s, pred_s in enumerate(pred):
        
        shape = pred_s.shape
        num_out  = int(shape[1] / num_anchor)
        pred_s = pred_s.view(shape[0], num_anchor, num_out, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        b, a, gj, gi = indices[s]  # img, anchor, gridy, gridx
        tobj = torch.zeros_like(pred_s[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            ps = pred_s[b, a, gj, gi] # N * (num_cls+5)

            # Regression (giou loss)
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[s]
            pbox = torch.cat((pxy, pwh), dim=1).to(device)  # predicted box
            giou = bbox_iou(pbox.T, tbox[s], x1y1x2y2=False, CIoU=True)
                        
            lbox += (1.0 - giou).mean()

            # Objectness
            gr = 1.0   # giou loss ratio (obj_loss = 1.0 or giou)
            tobj[b, a, gj, gi] = (1.0 - gr) + gr * giou.detach().clamp(0).type(tobj.dtype)

            # Classification
            if num_out - 5 > 1:  # only if multiple classes
                t = torch.full_like(ps[:, 5:], cn, device=device)
                t[range(n), tcls[s]] = cp
                lcls += BCEcls(ps[:, 5:], t)

        print (tobj.max())
        lobj += BCEobj(pred_s[..., 4], tobj) * balance[s]  # obj loss

    lbox *= 0.05
    lobj *= 1.0 * (1.4 if n_scale == 3 else 1.)
    lcls *= 0.5
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls)).detach()



def build_targets(pred, targets, anchors):
    '''
        pred     --->  [scale_1, scale2 ....]
        targets  --->  N*6 (image,class,x,y,w,h)
    '''
    tcls, tbox, indices, anch = [], [], [], []
    anchors = torch.Tensor(anchors).cuda()

    num_anchor, num_target = len(anchors[0]), targets.shape[0]
    
    ai = torch.arange(num_anchor, device=targets.device).float().view(num_anchor, 1).repeat(1, num_target)
    targets = torch.cat((targets.repeat(num_anchor, 1, 1), ai[:, :, None]), 2)
    # targets.shape --> 3*N*7, 3是每个尺度的anchor个数(将gt复制3份)

    g = 0.5
    offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g

    gain = torch.ones(7, device=targets.device)
    for s, stride in enumerate([16, 32]): #range(len(pred)): # 16, 32三个scale 
        
        gain[2:6] = torch.tensor(pred[s].shape)[[3, 2, 3, 2]] # (80,80,80,80)或 40或 20
        t = targets * gain  # 3*N*7

        anchor_s = anchors[s] / stride

        if num_target:
            r = t[:, :, 4:6] / anchor_s[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] <  4.0 # model.hyp['anchor_t']
            t = t[j]

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        b, c = t[:, :2].long().T  # img batch_id, class // shape --> N
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchor_s[a])
        tcls.append(c)  # class

    return tcls, tbox, indices, anch