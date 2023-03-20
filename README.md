# TLXCV
A Platform-agnostic Computer Vision Application Library, based on [TensorLayerX](https://github.com/tensorlayer/TensorLayerX). 

## Introduction
TLXCV  provides a set of algorithms and high-level APIs for computer vision tasks, such as image classification, object detection, semantic segmentation, etc.   
Some of the algorithms are converted from [PaddlePaddle](https://github.com/PaddlePaddle) implementations, and some are implemented from scratch.  

## 模型列表 Models
### 分类模型 Classification

| 序号 | 模型 | 类别误差 | 前后误差 | 状态 | 参考 |
| -- | -- | -- | -- | -- | -- |
| 1 | vgg16(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 2 | alexnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 3 | resnet50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 4 | resnet101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 5 | googlenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 6 | mobilenetv1(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 7 | mobilenetv2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 8 | mobilenetv3(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 9 | shufflenetv2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 10 | squeezenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 11 | inceptionv3(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 12 | regnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 13 | tnt(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 14 | darknet53(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 15 | densenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 16 | rednet50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 17 | rednet101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 18 | cspdarknet53(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 19 | efficientnet_b1(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 20 | efficientnet_b7(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 21 | dla34(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 22 | dla102(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 23 | dpn68(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 24 | dpn107(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 25 | ghostnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 26 | hardnet39(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 27 | hardnet85(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 28 | resnest50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 29 | resnext50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 30 | resnext101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 31 | rexnet(pretrained model) | 微小误差 | 0.00061244145 | 完成 | PaddleClas |
| 32 | se_resnext(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 33 | esnet_x0_5(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 34 | esnet_x1_0(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 35 | vit(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 36 | alt_gvt_small(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 37 | alt_gvt_base(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 38 | swin_transformer_base(pretrained model) | 0.0 |  |  | PaddleClas |
| 39 | swin_transformer_small(pretrained model) | 0.0 |  |  | PaddleClas |
| 40 | pcpvt_base(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 41 | pcpvt_large(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 42 | xception41(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 43 | xception65(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 44 | xception41_deeplab(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 45 | xception65_deeplab(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 46 | levit(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 47 | mixnet(pretrained model) | 微小误差 | 0.00048300158 | 完成 | PaddleClas |
| 48 | convnext(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 49 | cswin(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 50 | deittiny(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 51 | deitsmall(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 52 | deitbase(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 53 | dvt(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 54 | peleenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 55 | pp_hgnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 56 | pp_lcnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 57 | pp_lcnet_v2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 58 | pvt_v2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 59 | res2net(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |
| 60 | van(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |


### 分割模型 Segmentation

| 序号 | 模型 | 前后误差 | 状态 | 参考 |
| -- | -- | -- | -- | -- |
| 1 | fast_scnn | 0.0 | 完成 | PaddleSeg |
| 2 | hrnet | 0.0 | 完成 | PaddleSeg |
| 3 | encnet | 0.0 | 完成 | PaddleSeg |
| 4 | bisenet | 0.0 | 完成 | PaddleSeg |
| 5 | fastfcn | 0.0 | 完成 | PaddleSeg |
| 6 | enet | 0.0 | 完成 | PaddleSeg |


### 检测模型 Detection

| 序号 | 模型 | 前后误差 | 状态 | 方向 |
| -- | -- | -- | -- | -- |
| 1 | yolov3 | 0.0 | 完成 | PaddleDec |
| 2 | ssd | 0.0 | 完成 | PaddleDec |
| 3 | yolox | 0.0 | 完成 | PaddleDec |
| 4 | picodet_lcnet | 0.0 | 完成 | PaddleDec |
| 5 | fcos_r50 | 0.0 | 完成 | PaddleDec |
| 6 | fcos_dcn | 0.0 | 完成 | PaddleDec |
| 7 | RetinaNet | 0.0 | 完成 | PaddleDec |
| 8 | Mask_RCNN | 0.0 | 完成 | PaddleDec |
| 9 | Faster_RCNN | 0.0 | 完成 | PaddleDec |
| 10 | CascadeRCNN | 0.0 | 完成 | PaddleDec |
| 11 | SOLOv2 | 0.0 | 完成 | PaddleDec |
| 12 | GFL | 0.0 | 完成 | PaddleDec |
| 13 | TOOD | 0.0 | 完成 | PaddleDec |
| 14 | CenterNet | 0.0 | 完成 | PaddleDec |
| 15 | TTFNet | 0.0 | 完成 | PaddleDec |


### 遥感模型 Remote Sensing

| 序号 | 模型 | 前后误差 | 状态 | 参考 | 
| -- | -- | -- | -- | -- |
| 1 | bit | 0.0 | 完成 | PaddleRS |
| 2 | cdnet | 0.0 | 完成 | PaddleRS |
| 3 | stanet | 0.0 | 完成 | PaddleRS |
| 4 | fcef | 0.0 | 完成 | PaddleRS |
| 5 | fccdn | 0.0 | 完成 | PaddleRS |
| 6 | dsamnet | 0.0 | 完成 | PaddleRS |
| 7 | snunet | 0.0 | 完成 | PaddleRS |
| 8 | dsifn | 0.0 | 完成 | PaddleRS |
| 9 | unet | 0.0 | 完成 | PaddleRS |
| 10 | farseg | 0.0 | 完成 | PaddleRS |
| 11 | deeplab | 0.0 | 完成 | PaddleRS |
