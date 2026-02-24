# AP-PA (Adaptive-Patch-based Physical Attack)

## Modified Content 修改内容
Added support for ultralytics and mmdetection 3.x versions
增加了对ultralytics和mmdetection3.x版本的支持

## Usage Method 使用方法

Replace the `base_model.py` in the `mmdetection` folder with the `base_model.py` in the `model/base` directory of the `mmengine` library. 
Also, replace the `base.py` and `two_stage.py` in the `mmdetection` folder with the files of the same names in the `mmdet/models/detectors` directory of the `mmdet` you have installed.

将mmdetection文件夹中base_model.py替换mmengine库中model/base/base_model.py，将base.py和two_stage.py替换你安装的mmdet中mmdet/models/detectors/中的同名文件

## Introduction

Original author link:https://github.com/JiaweiLian/AP-PA

原作者链接：https://github.com/JiaweiLian/AP-PA

In this paper, a novel adaptive-patch-based physical attack (AP-PA) framework is proposed, which aims to generate adversarial patches that are adaptive in both physical dynamics and varying scales, and by which the particular targets can be hidden from being detected. Furthermore, the adversarial patch is also gifted with attack effectiveness against all targets of the same class with a patch outside the target (No need to smear targeted objects) and robust enough in the physical world. In addition, a new loss is devised to consider more available information of detected objects to optimize the adversarial patch, which can significantly improve the patch's attack efficacy (Average precision drop up to $87.86\%$ and $85.48\%$ in white-box and black-box settings, respectively) and optimizing efficiency. We also establish one of the first comprehensive, coherent, and rigorous benchmarks to evaluate the attack efficacy of adversarial patches on aerial detection tasks. We summarize our algorithm in [Benchmarking Adversarial Patch Against Aerial Detection](https://ieeexplore.ieee.org/document/9965436).

## Requirements:

* Pytorch 1.12

* Python 3.8

* MMDetection3.x

## Citation

If you use AP-PA method for attacks in your research, please consider citing

```
@article{lian2022benchmarking,
  title={Benchmarking Adversarial Patch Against Aerial Detection},
  author={Lian, Jiawei and Mei, Shaohui and Zhang, Shun and Ma, Mingyang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--16},
  year={2022},
  publisher={IEEE}
}
```
