# CED-Net-Crops-and-Weeds-Segmentation
The existing architectures for weeds and crops segmentation are quite deep, with millions of parameters that require longer training time. To overcome such limitations, we propose an idea of training small networks in cascade to obtain coarse-to-fine predictions, which are then combined to produce the final results.  Evaluation of the proposed network and comparison with other state-of-the-art networks are conducted using four publicly-available datasets: Rice seeding and weed dataset, BoniRob dataset, Carrot crop vs. weed dataset and paddy–millet dataset. The experimental results and their comparisons proclaim that the proposed network outperforms state-of-the-art architectures, such as U-Net, SegNet, FCN-8s, and DeepLabv3 over IoU, F1-score, sensitivity, true detection rate, and average precision comparison metrics, by utilizing only (1/5.74 × U-Net), (1/5.77 × SegNet), (1/3.04 × FCN-8s), and (1/3.24 × DeepLabv3)  fractions of total parameters. 
## This paper is published as a feature article in Electronics Journal and can be accesse here 
[CED-Net: Crops and Weeds Segmentation for Smart Farming](https://www.mdpi.com/2079-9292/9/10/1602)
## Overview
The proposed network architecture is shown in Figure below. The overall model training is performed
coarse weed prediction and Model-3 for crop prediction. The predictions of Model-1 and Model-3
are up-sampled, concatenated with corresponding input image size, and used as inputs by Model-2
and Model-4, respectively. Two cascaded networks (Model-1, Model-2) are thus trained for weed
predictions, and the other two (Model-3, Model-4) for crop predictions. In total, then, we have four
such small networks. The section that follows explains the network architecture and training details.
![Picture2](https://user-images.githubusercontent.com/56618776/98434177-c6985f00-2110-11eb-8104-25e618f122c3.png)
