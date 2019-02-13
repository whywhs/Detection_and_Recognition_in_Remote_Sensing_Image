# Detection_and_Recognition_in_Remote_Sensing_Image

## Notice 
This code is writting with reference to this [paper](https://arxiv.org/abs/1711.10398) by Hongyu Wang in Beihang University, and the code is built upon a fork of [Deformble Convolutional Networks](https://github.com/msracver/Deformable-ConvNets) and [Faster RCNN for DOTA](https://github.com/jessemelpolio/Faster_RCNN_for_DOTA).  

	@InProceedings{Xia_2018_CVPR,
	author = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
	title = {DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2018}
	}

In my code, I have tried different approaches mainly on the following points:  
1. Replace Faster_RCNN with [PANet](https://arxiv.org/abs/1803.01534).  
2. A new loss function called [Focal Loss](https://arxiv.org/abs/1708.02002) is attempted.    
 
## Requirements: Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet). We tested our code on [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60). Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. 

2. Python 2.7. We recommend using Anaconda2 to manage the environments and packages.

3. Some python packages: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running:
```
pip install Cython
pip install opencv-python==3.2.0.6
pip install easydict==1.6
```
4. For Windows users, Visual Studio 2015 is needed to compile cython module.


## Requirements: Hardware

Any NVIDIA GPUs with at least 4GB memory should be sufficient. 

## Installation

For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

## Preparation for Training & Testing

<!-- For R-FCN/Faster R-CNN\: -->

1. Please download [DOTA](https://captain-whu.github.io/DOTA/dataset.html) dataset, use the [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) to split the data into patches. And make sure the split images look like this:
```
./path-to-dota-split/images
./path-to-dota-split/labelTxt
./path-to-dota-split/test.txt
./path-to-dota-split/train.txt
```
The test.txt and train.txt are name of the subimages(without suffix) for train and test respectively.


2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), or [BaiduYun](https://pan.baidu.com/s/1YuB5ib7O-Ori1ZpiGf8Egw#list/path=%2F), or [Google drive](https://drive.google.com/open?id=1b6P-UMaBBpMPlcgvc38dMToPAa_Gyu6F), and put it under folder `./model`. Make sure it look like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	```

## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder  `./experiments/faster_rcnn/cfgs`.

2. Set the "dataset_path" and "root_path" in DOTA.yaml and DOTA_quadrangle.yaml. The "dataset_path" should be the father folder of "images" and "labelTxt". The "root_path" is the path you want to save the cache data.

3. Set the scales and aspect ratios as your wish in DOTA.yaml and DOTA_quadrangle.yaml.

4. To conduct experiments, run the python scripts with the corresponding config file as input. For example, train and test on quadrangle in an end-to-end manner, run
    ```
	python experiments/faster_rcnn/rcnn_dota_quadrangle_e2e.py --cfg experiments/faster_rcnn/cfgs/DOTA_quadrangle.yaml
    ```
    <!-- A cache folder would be created automatically to save the model and the log under `output/rfcn_dcn_coco/`. -->

## Experiment
![1](http://m.qpic.cn/psb?/V13MmUWH1KBoey/JnbnLwoALmmeEv172PDLBHh4s2KyvXSSd1rJ3zS0dzw!/b/dL8AAAAAAAAA&bo=VQMABFUDAAQRCT4!&rf=viewer_4)  

![1](http://m.qpic.cn/psb?/V13MmUWH1KBoey/SNLbUi4V6go.5MHB4tEtEGN61A.TK84hst*bxBGB8E0!/b/dL4AAAAAAAAA&bo=AAQABAAEAAQRKR4!&rf=viewer_4)  

![1](http://m.qpic.cn/psb?/V13MmUWH1KBoey/LgAe79Y468s0wL3OEZDVP7FeKcfjexSH*6YjkqLNNiY!/b/dFMBAAAAAAAA&bo=AAQABAAEAAQRKR4!&rf=viewer_4)  

![1](http://m.qpic.cn/psb?/V13MmUWH1KBoey/fCYMq76OLnWj.TlhnkHbaF.YO.7mT3exhNSxTD3IcQ0!/b/dLgAAAAAAAAA&bo=AAQABAAEAAQRKR4!&rf=viewer_4)
