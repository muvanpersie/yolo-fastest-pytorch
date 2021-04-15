# -*- coding=utf-8 -*-

import argparse
import sys

import glob
import cv2
import torch
import numpy as np

from models.yolo_fastest import YoloFastest


def export(args):
    from config.config import params

    io_params =  params["io_params"]
    names = io_params["names"]
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    model = YoloFastest(io_params).cuda()
    
    ckpt = torch.load(args.weights)
    model.load_state_dict(ckpt)

    dummy_input = torch.randn(1, 3, 256, 256).cuda()

    out_path = "output/output.onnx"
    
    torch.onnx.export(model, dummy_input, out_path, verbose=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',  type=str, default='output/bdd100k_epoch_81.pt', 
                        help="weights to load")

    parser.add_argument('--conf_thres', type=float, default=0.4)
    parser.add_argument('--save', default=False, action='store_true')    
    args = parser.parse_args()

    with torch.no_grad():
        export(args)
