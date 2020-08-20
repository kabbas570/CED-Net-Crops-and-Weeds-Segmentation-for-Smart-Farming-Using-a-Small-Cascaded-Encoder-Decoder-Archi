import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
import tensorflow as tf
import tensorflow.keras as    keras
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import numpy as np
import cv2
from Upsample_Conc import Upsample_Conc

### IMPORT aLL_MODELS ###
from Models import UNet,segnet,fcn_8,Deeplab_v3,MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4


### Import the Datset $$4
from Datasets.Paddy_Millet.paddy_millet_data import geneT,imgT # 'T' represents the Test data
from Datasets.BoniRob.bonirob_data import DataGenT
from Datasets.Carrot_Weed.carrot_data import DataGenT
from Datasets.Rice_Seeding.rice_seeding_data import DataGenT


### Import the Evaluation Metrics ###
from Evaluation_Metrics.Metric_ import Crop_iou,Weed_iou,mean_iou
from Evaluation_Metrics.True_Detection_Rate import metric_
from Evaluation_Metrics.mean_average import mean_AP





M4,M8=geneT() 
I4,I8=imgT()


I8,I4,M8,M4 = DataGenT()
M8=M8[:,:,:,:,0]

plt.figure()
plt.imshow(I8[10,:,:,:])


model=Deeplab_v3()
model.load_weights("DeeplabV3_paddy1.h5")
result = model.predict(I8,batch_size=2)
resultV=np.zeros([I8.shape[0],896,896,2])
resultV[np.where(result[:,:,:,:]>=.5)]=1


resultS=resultV
TDR=metric_(M8,resultS,10)
print('TDR at 10    :',TDR)
TDR=metric_(M8,resultS,15)
print('TDR at 15    :',TDR)
TDR=metric_(M8,resultS,20)
print('TDR at 20    :',TDR)

print('AP AT 10')
W_AP,R_AP,mAP=mean_AP(M8,resultS,10)
print('Mean Average Precision :',mAP)
print('Rice AP   :',R_AP)
print('Weed AP   :',W_AP)
print('AP AT 15')
W_AP,R_AP,mAP=mean_AP(M8,resultS,15)
print('Mean Average Precision :',mAP)
print('Rice AP   :',R_AP)
print('Weed AP   :',W_AP)
print('AP AT 20')
W_AP,R_AP,mAP=mean_AP(M8,resultS,20)
print('Mean Average Precision :',mAP)
print('Rice AP   :',R_AP)
print('Weed AP   :',W_AP) 

P1='/home/user01/data_ssd/Abbas/PAPER/Deeplab_results/paddy_millet/'

#I4,I8=imgT()
for i in range(40):
    IMG=I8[i,:,:,:].copy()  
    IMG[np.where(resultV[i,:,:,0]==1)]=[0,0,1]
    IMG[np.where(resultV[i,:,:,1]==1)]=[1,0,0]
    cv2.imwrite(os.path.join(P1 , str(i)+".png"),IMG*255)
    
    
    
    



model=fcn_8()
model.load_weights("_FCN_8s.h5")
result = model.predict(I8,batch_size=2)
resultF=np.zeros([I8.shape[0],896,896,2])
resultF[np.where(result[:,:,:,:]>=.5)]=1


model=UNet()
model.load_weights("_UNET14.h5")
result = model.predict(I8,batch_size=2)
resultU=np.zeros([I8.shape[0],896,896,2])
resultU[np.where(result[:,:,:,:]>=.5)]=1


model=segnet()
model.load_weights("BoniRob_SegNet.h5")
result = model.predict(I8,batch_size=2)
resultS=np.zeros([I8.shape[0],896,896,2])
resultS[np.where(result[:,:,:,:]>=.5)]=1

IoU_Crop=Crop_iou(resultS,M8)
IoU_Weed=Weed_iou(resultS,M8) 
IoU_Mean=mean_iou(resultS,M8) 



  
    

TDR=metric_(M8,resultS,10)
print('TDR at 10    :',TDR)
TDR=metric_(M8,resultS,15)
print('TDR at 15    :',TDR)
TDR=metric_(M8,resultS,20)
print('TDR at 20    :',TDR)


print('AP AT 10')
W_AP,R_AP,mAP=mean_AP(M8,resultS,10)
print('Mean Average Precision :',mAP)
print('Rice AP   :',R_AP)
print('Weed AP   :',W_AP)
print('AP AT 15')
W_AP,R_AP,mAP=mean_AP(M8,resultS,15)
print('Mean Average Precision :',mAP)
print('Rice AP   :',R_AP)
print('Weed AP   :',W_AP)
print('AP AT 20')
W_AP,R_AP,mAP=mean_AP(M8,resultS,20)
print('Mean Average Precision :',mAP)
print('Rice AP   :',R_AP)
print('Weed AP   :',W_AP)  




###  The propsed model is TESTED in Cascaded using MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4 ###
model1=MYMODEL1()
model2=MYMODEL2()
model3=MYMODEL3()
model4=MYMODEL4()
model1.load_weights("BoniRob_1.h5")
model2.load_weights("BoniRob_2.h5")
model3.load_weights("BoniRob_3.h5")
model4.load_weights("BoniRob_4.h5")

result1 = model1.predict(I4,batch_size=2)
I8 = np.float32(I8)
C_prediction1=Upsample_Conc(result1,I8)
resultW = model2.predict(C_prediction1,batch_size=2)
result2 = model3.predict(I4,batch_size=2)
C_prediction=Upsample_Conc(result2,I8)
resultR = model4.predict(C_prediction,batch_size=2)
resultw=np.zeros([I8.shape[0],896,896])
resultw[np.where(resultW[:,:,:,0]>=.6)]=1
resultr=np.zeros([I8.shape[0],896,896])
resultr[np.where(resultR[:,:,:,0]>=.6)]=1
PRE=[]
for i in range(62):
    re=np.zeros([896,896,2])
    W=resultw[i,:,:]
    R=resultr[i,:,:]
    re[:,:,0]=W
    re[:,:,1]=R
    PRE.append(re)
resultM=np.array(PRE)







    
    
model1=MYMODEL1()
model2=MYMODEL2()
model3=MYMODEL3()
model4=MYMODEL4()
model1.load_weights("BoniRob_1.h5")
model2.load_weights("BoniRob_2.h5")
model3.load_weights("BoniRob_3.h5")
model4.load_weights("BoniRob_4.h5")
result1 = model1.predict(I4,batch_size=2)
C_prediction1=Upsample_Conc(result1,I8)
resultW = model2.predict(C_prediction1,batch_size=2)
result2 = model3.predict(I4,batch_size=2)
C_prediction=Upsample_Conc(result2,I8)
resultR = model4.predict(C_prediction,batch_size=2)
PRE=[]
for i in range(len(I8)):
    re=np.zeros([896,896,2])
    R=resultR[i,:,:,0]
    W=resultW[i,:,:,0]
    r=np.zeros([896,896])
    w=np.zeros([896,896])
    w[np.where(W>=0.5)]=1
    r[np.where(R>=0.5)]=1
    r[np.where(w==1)]=0
    re[:,:,0]=r
    re[:,:,1]=w
    PRE.append(re)
result=np.array(PRE)




    







    
    








