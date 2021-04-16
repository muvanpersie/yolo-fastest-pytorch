# -*- coding=utf-8 -*-

import argparse
import sys

import glob
import cv2
import torch
import numpy as np

from models.yolo_fastest import YoloFastest
from utils.general import non_max_suppression, scale_coords

def resize_img(img0, new_shape=(736, 1280), color=(114, 114, 114)):
    shape = img0.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    unpad_shape = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - unpad_shape[0], new_shape[0] - unpad_shape[1]  # wh padding

    img = cv2.resize(img0, unpad_shape, interpolation=cv2.INTER_LINEAR)

    dw, dh = dw/2, dh/2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img


def detect(args):
    from config.config import params

    io_params =  params["io_params"]
    names = io_params["names"]
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    model = YoloFastest(io_params).cuda()
    
    ckpt = torch.load(args.weights)
    model.load_state_dict(ckpt)

    # warm up
    img = torch.zeros((1, 3, 640, 640)).cuda()
    _ = model(img)
    
    img_lists = glob.glob(args.root_dir + '*.jpg')
    for img_path in img_lists:
        print (img_path)
        
        img0 = cv2.imread(img_path)
        img = resize_img(img0, new_shape=(736, 1280))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).cuda().float() / 255.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        
        out = []
        strides = io_params["strides"]
        for i, pred_s in enumerate(pred):
            pred_s = pred_s[0] # 第一个batch

            (_, h, w) = pred_s.shape
            out_channel = io_params["num_cls"] + 5
            pred_s = pred_s.view(1, 3, out_channel, h, w).permute(0, 1, 3, 4, 2).contiguous()
             
            yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
            grid = torch.stack((xv, yv), 2).view((1, 1, h, w, 2)).float().to(pred_s.device)

            # bbox 
            anchors = io_params["anchors"][i]
            anchors = torch.tensor(anchors).float().view(3, -1, 2)
            anchors = anchors.clone().view(1, -1, 1, 1, 2).to(pred_s.device)

            pred_s = pred_s.sigmoid()
            pred_s[..., 0:2] = (pred_s[..., 0:2] * 2. - 0.5 + grid) * strides[i] # x,y
            pred_s[..., 2:4] = (pred_s[..., 2:4] * 2) ** 2 * anchors             # w,h

            out.append(pred_s.view(1, -1, out_channel))

        out = torch.cat(out, dim=1)
    
        # # output = [[x1,y1,x2,y2,conf,cls], [....]]   batch_size张图片的检测结果,放在list里面
        output = non_max_suppression(out, conf_thres=args.conf_thres)

        det = output[0] # detect阶段batch size为1
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            # draw results
            det = det.cpu()
            for i in range(det.shape[0]):
                x1, y1, x2, y2 = int(det[i][0]), int(det[i][1]), int(det[i][2]), int(det[i][3])
                conf = float(det[i][4])
                cls = int(det[i][5])
                
                cv2.rectangle(img0, (x1, y1), (x2, y2), color=colors[int(cls)], thickness=1)
                txt =  '%s %.2f' % (names[cls], conf)
                cv2.putText(img0, txt, (x1, y1-2), fontFace=0, fontScale=0.6, color=(225, 255, 255), lineType=cv2.LINE_AA)

        cv2.imshow("test", img0)
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/lance/data/DataSets/bdd/100k/images/test/')
    parser.add_argument('--weights',  type=str, default='output/epoch_99.pt', 
                        help="weights to load")

    parser.add_argument('--conf_thres', type=float, default=0.4)
    parser.add_argument('--save', default=False, action='store_true')    
    args = parser.parse_args()

    with torch.no_grad():
        detect(args)
