import argparse
import os
import time

import math 
import glob
import cv2
import torch
import numpy as np

from models.yolo_fastest import YoloFastest
from utils.general import non_max_suppression, scale_coords, plot_one_box


def detect(save_img=False):
    img_root_path = '/home/lance/data/DataSets/quanzhou/coco_style/cyclist/images/'
    weights = "output/epoch_6.pt" 
    imgsz = 640
    
    names = ["cyclist"]
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    device = torch.device('cuda:0')

    io_params =  { "num_cls" :  1,
                   "anchors" :  [[[30, 61],  [48, 65],  [52,132]], 
                                 [[52, 114], [114,199], [202,400]]] }
        
    # inference
    model = YoloFastest(io_params).to(device)
    
    ckpt = torch.load(weights)
    model.load_state_dict(ckpt)

    # warm up
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img)
    
    t0 = time.time()
    img_lists = sorted(glob.glob(img_root_path + '*.jpg'))
    for img_path in img_lists:
        
        img0 = cv2.imread(img_path)
        # img = letterbox(img0, new_shape=(960, 640))[0]
        img = cv2.resize(img0, (960,640), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # t1 = time_synchronized()
        pred = model(img)
        
        out = []
        scales = [16, 32]
        for i, pred_s in enumerate(pred):
            pred_s = pred_s[0] # 第一个batch

            (_, h, w) = pred_s.shape
            pred_s = pred_s.view(1, 3, 6, h, w).permute(0, 1, 3, 4, 2).contiguous()
             
            yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
            grid = torch.stack((xv, yv), 2).view((1, 1, h, w, 2)).float().to(pred_s.device)

            anchors = io_params["anchors"][i]
            anchors = torch.tensor(anchors).float().view(3, -1, 2)
            anchors = anchors.clone().view(1, -1, 1, 1, 2).to(pred_s.device)


            pred_s = pred_s.sigmoid()
            pred_s[..., 0:2] = (pred_s[..., 0:2] * 2. - 0.5 + grid) * scales[i] #x,y
            pred_s[..., 2:4] = (pred_s[..., 2:4] * 2) ** 2 * anchors            #w,h

            out.append(pred_s.view(1, -1, 6))

        out = torch.cat(out, dim=1)
    
        # # output = [[x1,y1,x2,y2,conf,cls], [....]]   batch_size张图片的检测结果,放在list里面 
        output = non_max_suppression(out, conf_thres=0.3)
        
        # t2 = time_synchronized()
        # print(" Infer time: {:.3f}".format(t2-t1))

        det = output[0]
        if det is not None and len(det):
            # 检测结果回归到原始图像
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # draw results
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

        cv2.imshow("test", img0)
        if cv2.waitKey(0) == ord('q'):  # q to quit
            raise StopIteration


if __name__ == '__main__':
   
    with torch.no_grad():
        detect()
