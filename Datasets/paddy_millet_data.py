import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
mask_id = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/Paddy_Millet/train/masks/*.png')):   # path to masks of train data
    mask_id.append(infile)
image_id = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/Paddy_Millet/train/images/*.png')):  # path to images of train data
    image_id.append(infile)
    
height=896
width=896
kernel = np.ones((5,5),np.uint8)
def gene():
    Mask=[]
    Mask1=[]
    for i in range(len(mask_id)):
        target=np.zeros([896,896,2])
        image = cv2.imread(mask_id[i],0)
        
        target[:,:,0][np.where(image==0)]=1       
        target[:,:,1][np.where(image==255)]=1
        target = cv2.morphologyEx(target, cv2.MORPH_OPEN, kernel)
        target1 = cv2.resize(target, (height//2,width//2), interpolation = cv2.INTER_AREA)
        target1[np.where(target1>=0.25)]=1
        target1 = cv2.morphologyEx(target1, cv2.MORPH_OPEN, kernel)
        
        Mask.append(target)  
        Mask1.append(target1)  
    Mask  = np.array(Mask)
    Mask1  = np.array(Mask1)
    return Mask1,Mask

def img():
    Mask=[]
    Mask1=[]
    for i in range(len(mask_id)):
        image = cv2.imread(image_id[i])
        image=image/255
        image1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        Mask.append(image)  
        Mask1.append(image1) 
    Mask  = np.array(Mask)
    Mask1  = np.array(Mask1)
      
    return Mask1,Mask


mask_idV = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/Paddy_Millet/valid/masks/*.png')):  # path to masks of validation data
    mask_idV.append(infile)
image_idV = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/Paddy_Millet/valid/images/*.png')):   # path to images of validation data
    image_idV.append(infile)

    
    
def geneV():
    Mask=[]
    Mask1=[]
    for i in range(len(image_idV)):
        target=np.zeros([896,896,2])
        image = cv2.imread(mask_idV[i],0)
        
        target[:,:,0][np.where(image==0)]=1       
        target[:,:,1][np.where(image==255)]=1
        target = cv2.morphologyEx(target, cv2.MORPH_OPEN, kernel)
        target1 = cv2.resize(target, (height//2,width//2), interpolation = cv2.INTER_AREA)
        target1[np.where(target1>=0.25)]=1
        target1 = cv2.morphologyEx(target1, cv2.MORPH_OPEN, kernel)
        
        Mask.append(target)  
        Mask1.append(target1)  
    Mask  = np.array(Mask)
    Mask1  = np.array(Mask1)

    return Mask1,Mask




def imgV():
    Mask=[]
    Mask1=[]
    for i in range(len(image_idV)):
        image = cv2.imread(image_idV[i])
        image=image/255
        image1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        Mask.append(image)  
        Mask1.append(image1) 
    Mask  = np.array(Mask)
    Mask1  = np.array(Mask1)
      
    return Mask1,Mask

import glob
mask_idT = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/Paddy_Millet/test/masks/*.png')):    # path to masks of test data
    mask_idT.append(infile)
image_idT = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/PAPER/Datasets/Paddy_Millet/test/images/*.png')):  # path to images of test data
    image_idT.append(infile)  
def geneT():
    Mask=[]
    Mask1=[]
    for i in range(len(mask_idT)):
        target=np.zeros([896,896,2])
        image = cv2.imread(mask_idT[i],0)
        
        target[:,:,0][np.where(image==0)]=1       
        target[:,:,1][np.where(image==255)]=1
        target = cv2.morphologyEx(target, cv2.MORPH_OPEN, kernel)
        target1 = cv2.resize(target, (height//2,width//2), interpolation = cv2.INTER_AREA)
        target1[np.where(target1>=0.25)]=1
        target1 = cv2.morphologyEx(target1, cv2.MORPH_OPEN, kernel)
        
        Mask.append(target)  
        Mask1.append(target1)  
    Mask  = np.array(Mask)
    Mask1  = np.array(Mask1)
    return Mask1,Mask
def imgT():
    Mask=[]
    Mask1=[]
    for i in range(len(mask_idT)):
        image = cv2.imread(image_idT[i])
        image=image/255
        image1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        Mask.append(image)  
        Mask1.append(image1) 
    Mask  = np.array(Mask)
    Mask1  = np.array(Mask1)    
    return Mask1,Mask


