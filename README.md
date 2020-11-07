## CED-Net-Crops-and-Weeds-Segmentation
Convolutional neural networks (CNNs) have achieved state-of-the-art performance in numerous aspects of human life and the agricultural sector is no exception. One of the main objectives of deep learning for smart farming is to identify the precise location of weeds and crops on farmland. In this paper, we propose a semantic segmentation method based on a cascaded encoder-decoder network to differentiate weeds from crops. The existing architectures for weeds and crops segmentation are quite deep, with millions of parameters that require longer training time. To overcome such limitations, we propose an idea of training small networks in cascade to obtain coarse-to-fine predictions, which are then combined to produce the final results.  Evaluation of the proposed network and comparison with other state-of-the-art networks are conducted using four publicly-available datasets: Rice seeding and weed dataset, BoniRob dataset, Carrot crop vs. weed dataset and paddy–millet dataset. The experimental results and their comparisons proclaim that the proposed network outperforms state-of-the-art architectures, such as U-Net, SegNet, FCN-8s, and DeepLabv3 over IoU, F1-score, sensitivity, true detection rate, and average precision comparison metrics, by utilizing only (1/5.74 × U-Net), (1/5.77 × SegNet), (1/3.04 × FCN-8s), and (1/3.24 × DeepLabv3)  fractions of total parameters. 
