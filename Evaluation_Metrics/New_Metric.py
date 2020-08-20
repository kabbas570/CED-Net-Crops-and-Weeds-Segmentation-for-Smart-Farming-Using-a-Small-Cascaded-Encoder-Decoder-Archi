import numpy as np
import cv2

def give_centers(T,P):
    target_centers=[]
    predicted_centers=[]

    T = T.astype(np.uint8)
    cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])  
        cY = int(M["m01"] / M["m00"])
        target_centers.append((cX,cY))
        
    P = P.astype(np.uint8)
    cnts1 = cv2.findContours(P, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]
    for c in cnts1:
        M = cv2.moments(c)
        if (M["m00"]==0):
            M["m00"]=1
        cX1 = int(M["m10"] / M["m00"])  
        cY1 = int(M["m01"] / M["m00"])
        predicted_centers.append((cX1,cY1))
    return target_centers,predicted_centers

def euclidean(list_org, list_pred):
    difference=list_org-list_pred
    euclidean_distance=np.sqrt(np.square(difference[:,0])+np.square(difference[:,1]))
    return euclidean_distance
def TP_FP(T,P,TH):
    TP=[]
    FP=[]
    Acc_TP=[]
    Acc_FP=[]
    pix_thresh=TH
    for index in range(40):
        target_centers,predicted_centers=give_centers(T[index,:,:],P[index,:,:]) 
        #print(target_centers)
        if len(predicted_centers)==len(target_centers):
            for b in predicted_centers:
                single_value=min(euclidean(np.array([b]), target_centers))
                #print(single_value)
                if single_value<=pix_thresh:
                    TP.append(1)
                    FP.append(0)
                    Acc_TP.append(sum(TP))
                    Acc_FP.append(sum(FP))                             
                else:
                    FP.append(1)
                    TP.append(0)
                    Acc_FP.append(sum(FP))
                    Acc_TP.append(sum(TP))         
        
        if len(predicted_centers)<len(target_centers):
            for b in predicted_centers:
                single_value=min(euclidean(np.array([b]), target_centers))
                if single_value<=pix_thresh:
                    TP.append(1)
                    FP.append(0)
                    Acc_TP.append(sum(TP))
                    Acc_FP.append(sum(FP))                 
                else:
                    FP.append(1)         
                    TP.append(0)
                    Acc_FP.append(sum(FP))
                    Acc_TP.append(sum(TP))
        if len(predicted_centers)>len(target_centers):
            for b in target_centers:
                single_value=min(euclidean(np.array([b]), predicted_centers))
                if single_value<=pix_thresh:
                    TP.append(1)
                    FP.append(0)
                    Acc_TP.append(sum(TP))
                    Acc_FP.append(sum(FP))
                    
                else:
                    FP.append(1)
                    TP.append(0)
                    Acc_FP.append(sum(FP))
                    Acc_FP.append(sum(FP))               
            FP.append((len(predicted_centers)-len(target_centers)))
            TP.append(0)
            Acc_FP.append(sum(FP))
            Acc_TP.append(sum(TP))      
    return TP,FP 