## Pytorch version yolo-fastest

This repo is base on [Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) and [yolov5](https://github.com/ultralytics/yolov5). It can do one class detection task now, and it will be updated! 

### 1. Requirements
- Ubunut 16+
- CUDA 10.2+
- pytorch 1.6.0+
- opencv-python==4.2.0.34

### 2. Train
Dataset label is in MSCOCO style, (class, x_c, y_c, w, h).
```
    --DATA_ROOT
        |
        |---images/           # all image files
        |
        |---labels/           # annotation files
        |
        |---labels.cache      # generated when run train.py
```

Change `params[io_params][train_path]` to your own `DATA_ROOT`, then run the command:
```
python train.py
```

There are two mode for data preparation:
- Rect mode 
 > Scale image to a fixed size and keep its aspect ration with padding.
- Mosaic mode
 > Use Mosaic augment as yolov5. 


### 3. Test
```
python detect.py --
```