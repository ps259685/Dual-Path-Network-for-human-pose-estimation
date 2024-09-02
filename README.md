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
