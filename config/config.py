
'''
# for cyclist
params = {
        "io_params": {
            "save_path": 'output',
            "train_path": '/home/lance/data/DataSets/quanzhou/coco_style/cyclist',
            "input_size": 640,
            "num_cls":  1,
            "names": ['cyclist'],
            "anchors":  [[[12, 18],  [37, 49],  [52,132]], 
                         [[115, 73], [119,199], [242,238]]],
            "strides": [16, 32],
        },

        "augment_params": {
            "aug_mode": "mosaic", # "mosaic" or "rect"
            "mosaic_size": 640,   # only valid for mosaic mode
            "new_shape": (736, 1280), # only vaild for rect mode
            "hsv_h": 0.015,     # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,       # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,       # image HSV-Value augmentation (fraction)
            "degrees": 0.0,     # image rotation (+/- deg)
            "translate": 0.0,   # image translation (+/- fraction)
            "scale": 1.0,       # image scale (+/- gain)
            "shear": 0.0,       # image shear (+/- deg)
            "perspective": 0.0, # image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,      # image flip up-down (probability)
            "fliplr": 0.5,      # image flip left-right (probability)
            "mixup": 0.0,       # image mixup (probability)
        },

        "train_params": {
            "total_epochs": 100,
            "batch_size": 24,
            "lr0": 0.001,         # initial learning rate (SGD=1E-2, Adam=1E-3)
            "momentum": 0.937,    # SGD momentum/Adam beta1
            "weight_decay": 0.0005,
        },
    }

'''


'''
# for ms coco
params = {
        "io_params": {
            "save_path": 'output',
            "train_path": '/home/lance/data/DataSets/quanzhou/coco_style/cyclist',
            "input_size": 640,
            "num_cls":  80,
            "names": [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                       'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                       'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                       'hair drier', 'toothbrush' ],
            "anchors":  [[[12, 18],  [37, 49],  [52,132]], 
                         [[115, 73], [119,199], [242,238]]],
            "strides": [16, 32],
        },

        "augment_params": {
            "aug_mode": "mosaic", # "mosaic" or "rect"
            "mosaic_size": 640,   # only valid for mosaic mode
            "new_shape": (736, 1280), # only vaild for rect mode
            "hsv_h": 0.015,     # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,       # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,       # image HSV-Value augmentation (fraction)
            "degrees": 0.0,     # image rotation (+/- deg)
            "translate": 0.0,   # image translation (+/- fraction)
            "scale": 1.0,       # image scale (+/- gain)
            "shear": 0.0,       # image shear (+/- deg)
            "perspective": 0.0, # image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,      # image flip up-down (probability)
            "fliplr": 0.5,      # image flip left-right (probability)
            "mixup": 0.0,       # image mixup (probability)
        },

        "train_params": {
            "total_epochs": 100,
            "batch_size": 24,
            "lr0": 0.001,         # initial learning rate (SGD=1E-2, Adam=1E-3)
            "momentum": 0.937,    # SGD momentum/Adam beta1
            "weight_decay": 0.0005,
        },
    }
'''



# for bdd100k
params = {
        "io_params": {
            "save_path": 'output',
            "train_path": '/home/lance/data/DataSets/bdd/100k',
            "input_size": 640,
            "num_cls":  7,
            "names": ['car', 'truck', 'van', 'bus', 'pedestrian', 'cyclist', 'cone'],
            "anchors":  [[[12, 18],  [37, 49],  [52,132]], 
                         [[115, 73], [119,199], [242,238]]],
            "strides": [16, 32],
        },

        "augment_params": {
            "aug_mode": "rect", # "mosaic" or "rect"
            "mosaic_size": 640,   # only valid for mosaic mode
            "new_shape": (736, 1280), # only vaild for rect mode
            "hsv_h": 0.015,     # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,       # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,       # image HSV-Value augmentation (fraction)
            "degrees": 0.0,     # image rotation (+/- deg)
            "translate": 0.0,   # image translation (+/- fraction)
            "scale": 1.0,       # image scale (+/- gain)
            "shear": 0.0,       # image shear (+/- deg)
            "perspective": 0.0, # image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,      # image flip up-down (probability)
            "fliplr": 0.5,      # image flip left-right (probability)
            "mixup": 0.0,       # image mixup (probability)
        },

        "train_params": {
            "total_epochs": 100,
            "batch_size": 12,
            "lr0": 0.001,         # initial learning rate (SGD=1E-2, Adam=1E-3)
            "momentum": 0.937,    # SGD momentum/Adam beta1
            "weight_decay": 0.0005,
        },
    }
