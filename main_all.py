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


#      ### ALL MODELS ARE IMPORTED HERE ###
from Models import UNet,segnet,fcn_8,Deeplab_v3,MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4


#    ### THE DATASET TO BE USED IN THE EXPERIMENTS ###

from Datasets.BoniRob.bonirob_data import DataGenV,DataGen    # 'V' represents the validation data
from Datasets.Rice_Seeding.rice_seeding_data import DataGenV,DataGen    # 'V' represents the validation data
from Datasets.Carrot_Weed.carrot_data import DataGenV,DataGen  # 'V' represents the validation data
from Datasets.Paddy_Millet.paddy_millet_data import gene,img,geneV,imgV # 'V' represents the validation data

#    ### LOSS FUNCTION AND THE METRIC TO BE OPTIMIZED IN TRAINING ###
from Dice_Loss import dice_coef,dice_coef_loss

### Utility FUnction for upsampling and concatinating the feature maps and images   ###
from Upsample_Conc import Upsample_Conc

train_genx,C1,train_geny,Y1 = DataGen()        # train_genx_Size= 896 x 896 x 3 / train_geny_Size= 896 x 896 x 1
train_genxV,C1V,train_genyV,Y1V = DataGenV()   # C1_Size = 448 x 448 x 3 / Y1_Size= 448 x 448 x 1


M8=train_geny[:,:,:,:,0]
M8V=train_genyV[:,:,:,:,0]

M4,M8=gene()
I4,I8=img()

M4V,M8V=geneV()
I4V,I8V=imgV()



#    ### PREPRATION OF DATA FOR MODELS TO MATCH THE INPUT/OUTPUT DIMENSSIONS ###
a=train_geny[:,:,:,0]
b=train_geny[:,:,:,1]
aV=train_genyV[:,:,:,0] 
bV=train_genyV[:,:,:,1]
a1=Y1[:,:,:,0]
b1=Y1[:,:,:,1]
a1V=Y1V[:,:,:,0]
b1V=Y1V[:,:,:,1]

M8=train_geny[:,:,:,:,0]
M8V=train_genyV[:,:,:,:,0]
epochs=40
Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)






model=UNet()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(train_genx,M8,validation_data=(train_genxV, M8V),batch_size=2, 
                    epochs=epochs)
model.save_weights("Carrot_Unet.h5")

model=segnet()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(train_genx,M8,validation_data=(train_genxV,M8V),batch_size=2, 
                    epochs=epochs)
model.save_weights("Carrot_Seg.h5")

model=fcn_8()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])

model.fit(train_genx,M8,validation_data=(train_genxV,M8V),batch_size=2, 
                    epochs=epochs)
model.save_weights("FCN_Carrot.h5")



model=Deeplab_v3()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(train_genx,M8,validation_data=(train_genxV, M8V),batch_size=2, 
                    epochs=epochs)
model.save_weights("DeeplabV3_paddy6.h5")



  ###          The propsed model is trained in Cascaded using MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4  ###


##      Level-1 Models (Model-1)     ##
model=MYMODEL1()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C1,b1,validation_data=(C1V, b1V),batch_size=2,
                    epochs=epochs)
model.save_weights("Carrot_1.h5")
result1 = model.predict(C1)
result1V = model.predict(C1V)
C_prediction=Upsample_Conc(result1,train_genx)
CV_prediction=Upsample_Conc(result1V,train_genxV)

##      Level-2 Models (Model-3)      ##
model=MYMODEL2()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C_prediction,b,validation_data=(CV_prediction, bV),batch_size=2,
                    epochs=epochs)
model.save_weights("Carrot_2.h5")
resultW = model.predict(C_prediction)
resultVW = model.predict(CV_prediction)


##      Level-1 Models (Model-2)      ##
model=MYMODEL3()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C1,a1,validation_data=(C1V, a1V),batch_size=2,
                    epochs=epochs)
model.save_weights("Carrot_3.h5")
result2 = model.predict(C1)
result2V = model.predict(C1V)
C_prediction=Upsample_Conc(result2,train_genx)
CV_prediction=Upsample_Conc(result2V,train_genxV)

##       Level-2 Models (Model-4)     ##
model=MYMODEL4()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C_prediction,a,validation_data=(CV_prediction, aV),batch_size=2,
                    epochs=epochs)
resultR = model.predict(C_prediction)
resultVR = model.predict(CV_prediction)
model.save_weights("Carrot_4.h5")


