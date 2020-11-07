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
## Rsults and Comparisons
###Rice Seeding and Weed Segmentation Dataset
![Figure_4](https://user-images.githubusercontent.com/56618776/98434314-e54b2580-2111-11eb-98bd-268b92bba7c6.png)

###BoniRob Dataset

![Figure_5](https://user-images.githubusercontent.com/56618776/98434315-e7ad7f80-2111-11eb-946d-d73e8c72b588.png)

###Carrot Crop and Weed

![Figure_6](https://user-images.githubusercontent.com/56618776/98434317-e9774300-2111-11eb-9640-6e1fd8d85e02.png)

###Paddy-Millet Datase

![Figure_7](https://user-images.githubusercontent.com/56618776/98434321-ebd99d00-2111-11eb-827b-9f06aa6f62b9.png)


#Citation Request
If you use CED-Net in your project, please cite the following paper

@article{https://www.mdpi.com/2079-9292/9/10/1602,
  title={CED-Net: Crops and Weeds Segmentation for Smart
Farming Using a Small Cascaded
Encoder-Decoder Architecture},
  author={Abbas Khan,Talha Ilyas Muhammad Umraiz, Zubaer Ibna Mannan and
Hyongsuk Kim },
  journal={MDPI Electronics},
  volume={9},
 
  year={2020},
  publisher={Elsevier}
}
