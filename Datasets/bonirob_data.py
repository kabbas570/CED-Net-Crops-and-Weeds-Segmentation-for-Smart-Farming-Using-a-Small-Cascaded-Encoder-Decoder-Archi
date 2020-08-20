import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
mask_id = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/BoniRob/train/masks/*.png')): # path to masks of train data
    mask_id.append(infile)
image_ = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/BoniRob/train/images/*.png')):  # path to images of train data
    image_.append(infile)
mask_V = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/BoniRob/valid/masks/*.png')):   # path to masks of validation data
    mask_V.append(infile)
image_V = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/BoniRob/valid/images/*.png')): # path to images of validation data
    image_V.append(infile)
mask_T = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/BoniRob/test/masks/*.png')):   # path to masks of test data
    mask_T.append(infile)
image_T = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/BoniRob/test/images/*.png')):  # path to images of test data
    image_T.append(infile)  
      



height=896
width=896
def DataGen():  
    img_ = []
    mask_  = []
    c1=[]
    y1=[]
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        cc1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)     
        mask = cv2.imread(mask_id[i],0)
        mask[np.where(mask==0)]=1
        target=np.zeros([966,1296,2])
        target[:,:,0][np.where(mask==149)]=1
        target[:,:,1][np.where(mask==76)]=1

        mask = cv2.resize(target, (height,width), interpolation = cv2.INTER_AREA)
        yy1 = cv2.resize(target, (height//2,width//2), interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=-1)
        yy1 = np.expand_dims(yy1, axis=-1)
        img_.append(image)
        mask_.append(mask)
        c1.append(cc1)
        y1.append(yy1)   
    img_ = np.array(img_)
    C1=np.array(c1)
    Y1=np.array(y1)
    mask_  = np.array(mask_)
    mask_[np.where(mask_!=0)]=1
    Y1[np.where(Y1!=0)]=1
    return img_,C1,mask_,Y1
def DataGenV():  
    img_ = []
    mask_  = []
    c1=[]
    y1=[]
    for i in range(len(image_V)):
        image = cv2.imread(image_V[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        cc1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)     
        mask = cv2.imread(mask_V[i],0)
        mask[np.where(mask==0)]=1
        target=np.zeros([966,1296,2])
        target[:,:,0][np.where(mask==149)]=1
        target[:,:,1][np.where(mask==76)]=1

        mask = cv2.resize(target, (height,width), interpolation = cv2.INTER_AREA)
        yy1 = cv2.resize(target, (height//2,width//2), interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=-1)
        yy1 = np.expand_dims(yy1, axis=-1)
        img_.append(image)
        mask_.append(mask)
        c1.append(cc1)
        y1.append(yy1)

    img_ = np.array(img_)
    C1=np.array(c1)
    Y1=np.array(y1)
    mask_  = np.array(mask_)
    mask_[np.where(mask_!=0)]=1
    Y1[np.where(Y1!=0)]=1
    return img_,C1,mask_,Y1
def DataGenT():  
    img_ = []
    mask_  = []
    c1=[]
    y1=[]
    for i in range(len(image_T)):
        image = cv2.imread(image_T[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        cc1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)     
        mask = cv2.imread(mask_T[i],0)
        mask[np.where(mask==0)]=1
        target=np.zeros([966,1296,2])
        target[:,:,0][np.where(mask==149)]=1
        target[:,:,1][np.where(mask==76)]=1

        mask = cv2.resize(target, (height,width), interpolation = cv2.INTER_AREA)
        yy1 = cv2.resize(target, (height//2,width//2), interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=-1)
        yy1 = np.expand_dims(yy1, axis=-1)
        img_.append(image)
        mask_.append(mask)
        c1.append(cc1)
        y1.append(yy1)
    img_ = np.array(img_)
    C1=np.array(c1)
    Y1=np.array(y1)
    mask_  = np.array(mask_)
    mask_[np.where(mask_!=0)]=1
    Y1[np.where(Y1!=0)]=1
    return img_,C1,mask_,Y1

