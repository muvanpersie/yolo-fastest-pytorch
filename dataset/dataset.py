# -*-coding=utf-8 -*-

import glob
import math
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def get_hash(files):
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


class SimpleDataset(Dataset):
    def __init__(self, path, img_size=640, augment=True, aug_params=None, rect=False, pad=0.0):

        self.img_files = sorted(glob.glob(path+"/images/train2014/*.jpg"))
        self.label_files = [x.replace('images', 'labels').replace('jpg', 'txt') for x in self.img_files]
  
        self.img_size = img_size
        self.augment = augment
        self.rect = rect

        self.aug_params = aug_params

        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        
        
        #cache为字典 cache[img_path] = [annos, shape]
        cache_path = path + '/labels.cache'
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)
            if cache['hash'] != get_hash(self.label_files + self.img_files):
                cache = self.cache_labels(cache_path)
        else:
            cache = self.cache_labels(cache_path)

        annos, shapes = zip(*[cache[x] for x in self.img_files])
        self.annos = list(annos)
        # self.shapes = np.array(shapes, dtype=np.float64) #图片的原始尺寸

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        if self.mosaic:
            img, labels = load_mosaic(self, index)
        else:
            img, labels = load_rect(self, index, new_shape=(736, 1280))

        # 此时 labels为绝对大小

        if len(labels):
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                labels[:, [2, 4]] /= img.shape[0]
                labels[:, [1, 3]] /= img.shape[1]
        
        if self.augment:
            # border = self.mosaic_border if self.mosaic else (0, 0)
            # img, labels = random_perspective(img, labels,
            #                                  degrees=self.aug_params['degrees'],
            #                                  translate=self.aug_params['translate'],
            #                                  scale=self.aug_params['scale'],
            #                                  shear=self.aug_params['shear'],
            #                                  perspective=self.aug_params['perspective'],
            #                                  border=border)

            # colorspace
            augment_hsv(img)

            # flip
            if random.random() < self.aug_params['fliplr']:
                img = np.fliplr(img)
                labels[:, 1] = 1 - labels[:, 1]

        nL = len(labels)  # number of labels
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out

    def cache_labels(self, path='labels.cache'):
        x = {}
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        
        for (img_path, label_path) in pbar:
            annos = []
            img = cv2.imread(img_path)
            if img is not None:
                shape = (img.shape[1], img.shape[0])
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    annos = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                if len(annos) == 0:
                    annos = np.zeros((0, 5), dtype=np.float32)
            else:
                annos = np.zeros((0, 5), dtype=np.float32)
            
            x[img_path] = [annos, shape]

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)
        
        return x

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0)


# 保持原图的宽高比进行缩放, 两边填充
def load_rect(self, index, new_shape=(640, 960), color=(114, 114, 114)):

    img_path = self.img_files[index]
    img = cv2.imread(img_path)
    labels = self.annos[index] # xywh 0-1之间

    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    unpad_shape = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - unpad_shape[0], new_shape[0] - unpad_shape[1]  # wh padding
    
    img = cv2.resize(img, unpad_shape, interpolation=cv2.INTER_LINEAR)
    
    dw, dh = dw/2, dh/2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    if len(labels):
        labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])
        labels[:, [2, 4]] = (labels[:, [2, 4]] * shape[0]) * r + top
        labels[:, [1, 3]] = (labels[:, [1, 3]] * shape[1]) * r + left

    return img, labels


# loads 1 image from dataset, returns img, original hw, resized hw
def load_image(self, index): 
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size

    if r != 1:
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def load_mosaic(self, index):
    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.annos) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.annos[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    return img4, labels4


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


if __name__ == "__main__":
    from torch.utils.data import DataLoader, BatchSampler

    train_path = '/home/lance/data/DataSets/quanzhou/coco_style/cyclist'
    augment_params = {
            "hsv_h": 0.015,    # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,      # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,      # image HSV-Value augmentation (fraction)
            "degrees": 0.0,    # image rotation (+/- deg)
            "translate": 0.1,  # image translation (+/- fraction)
            "scale": 0.1,      # image scale (+/- gain)
            "shear": 0.0,      # image shear (+/- deg)
            "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,     # image flip up-down (probability)
            "fliplr": 0.5,     # image flip left-right (probability)
            "mixup": 0.0,      # image mixup (probability)
            }
    
    dataset = SimpleDataset(train_path, img_size=640, aug_params=augment_params, rect=True)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, sampler=None,
                            pin_memory=True, collate_fn=SimpleDataset.collate_fn)


    for batch_data in dataloader:
        print ("this is test!")




