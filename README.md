# Dual-Path-Network-for-human-pose-estimation
This is a pytorch implement of Dual-Path Network for pose estimation.

>Abstract—Pose estimation plays a important role in com
puter vision, with wide-ranging applications such as human
computer interaction(HCI), motion analysis, and Virtual real
ity(VR). Recently, many researchers have used Convolutional
 Neural Networks (CNNs) as the foundation for their models.
 However, traditional CNNs struggle to effectively capture multi
scale information. Although increasing the depth of these net
works could address this issue, it simultaneously leads to a rise
 in the number of parameters and computational complexity.
 This growing complexity and parameter count present additional
 challenges. To tackle this, we propose a multi-branch module
 designed to capture multi-scale information while keeping pa
rameters and computational complexity low. Additionally, we
 incorporate channel attention and spatial attention mechanisms
 into the module without significantly increasing parameter or
 computational burdens.

## Architecture of Dual-Path Network
![image](https://github.com/ps259685/Dual-Path-Network-for-human-pose-estimation/blob/main/DPN/figures/Overall%20Architecture.jpg)
## Multi-Branch Step Asymmetric Conv(MBSAC) Block
![image](https://github.com/ps259685/Dual-Path-Network-for-human-pose-estimation/blob/main/DPN/figures/Multi-Branch%20Step%20Asymmertic%20Conv(MBSAC)%20Block.jpg)
## Multi-Branch Inverted Residual Asymmetric Conv(MBIRAC) Block
![image](https://github.com/ps259685/Dual-Path-Network-for-human-pose-estimation/blob/main/DPN/figures/Multi-Branch%20Inverted%20Residual%20Asymmetric%20Conv(MBIRAC)%20Block.jpg)
##The attention module, Pose Feature Extractor
![image](https://github.com/ps259685/Dual-Path-Network-for-human-pose-estimation/blob/main/DPN/figures/The%20attention%20module%2C%20Pose%20Feature%20Extractor.jpg)

## Repo Structure
```
$DPN_ROOT
|-- exps
|   |-- exp1
|   |-- exp2
|   |-- ...
|
|-- datasets
|   |-- COCO
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |       |-- train2014
|   |       |-- val2014
|   |
|   |-- MPII
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |
|   |-- OCHuman
|   |   |-- images
|   |
|   |-- HumanArt
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |
|   |-- ExLPose
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |
|-- model_logs
|   
|-- lib
|   |-- models
|   |-- utils
|
|-- cvpack
|
|-- README
```

## Quick Start
### Installation
1. Install Pytorch from the [Pytorch website](https://pytorch.org/).
2. Download this repo
3. Install requirements
4. Install COCOAPI from [cocoapi official github](https://github.com/cocodataset/cocoapi).

### Train
Go to the exp folder, e.g.
`cd DPN_ROOT/exps/DPN.coco`
To run:
`python config.py -log`
`python -m train.py`
To test:
`python -m test.py -i iter_num`
_iter_num is the iteration number which you want to test._
