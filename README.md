# Pytorch version yolo-fastest

This repo is base on [Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) and [yolov5](https://github.com/ultralytics/yolov5). It is now can be used for BDD100k and Mscoco dataset now.

## 1. Requirements
- Ubunut 16+
- CUDA 10.2+
- pytorch 1.6.0+
- opencv-python==4.2.0.34

## 2. Train

### 2.1 Data preparation
Prepare your dataset structure as follows: 

```
    --DATA_ROOT
        |
        |---images/           # all image files
        |     |---000000.jpg
        |          .....
        |
        |---labels/           # annotation files
        |     |---000000.txt 
        |          .....
        |
        |---labels.cache      # generated when run train.py
```

Annotaion file is in `(class, x_c, y_c, w, h)` format. An example is as follows:
```
0 0.283203 0.108750 0.072656 0.217500 
1 0.346875 0.068125 0.056250 0.136250 
```

`x_c` and `y_c` means the center of object bounding box, while `w` and `h` means the width and height of object bounding box. All of them are normalized by the image's width and height.

### 2.2 Modify configuration

Modify paramteres in `config/config.py` according to your task:

There are two mode for dataloader, and it can be set in `augment_params["aug_mode"]`
- rect mode 
 > Load one image and scale it to `new_shape` while keep its aspect ratio with padding.
- mosaic mode
 > Use Mosaic augment as yolov5. Load four images once and mix them as one. 

And then run the command to train.
```
python train.py
```

## 3. Detect
```
python detect.py --root_dir YOUR_IMAGE_SET_PATH --weights output/epoch_xxx.pt --conf_thres 0.4
```
Press `Esc` to quit. 

## 4. Evalution

to be continued...