from Evaluation_Metrics.Average_Precision import ElevenPointInterpolatedAP
from Evaluation_Metrics.New_Metric import TP_FP

def mean_AP(GT,PRED,TH):
    Rice_GT=GT[:,:,:,1]
    Weed_GT=GT[:,:,:,0]
    Rice_P=PRED[:,:,:,1]
    Weed_P=PRED[:,:,:,0]
    TP_R,FP_R=TP_FP(Rice_GT,Rice_P,TH)
    Acc_TPR=[]
    s=0
    for i in TP_R:
        s=i+s
        Acc_TPR.append(s)
        
    Acc_FPR=[]
    s=0
    for i in FP_R:
        s=i+s
        Acc_FPR.append(s)
        
    precR=[]
    recR=[]
    for i, j in zip(Acc_TPR, Acc_FPR):
        C=i/(i+j)
        precR.append(C)
        recR.append(i/238)
    TP_W,FP_W=TP_FP(Weed_GT,Weed_P,TH)
    Acc_TPW=[]
    s=0
    for i in TP_W:
        s=i+s
        Acc_TPW.append(s)       
    Acc_FPW=[]
    s=0
    for i in FP_W:
        s=i+s
        Acc_FPW.append(s)
        
    precW=[]
    recW=[]
    for i, j in zip(Acc_TPW, Acc_FPW):
        C=i/(i+j)
        precW.append(C)
        recW.append(i/93)
    AP1= ElevenPointInterpolatedAP(recW, precW)
    AP2= ElevenPointInterpolatedAP(recR, precR)
    mAP=(AP1+AP2)/2
    return AP1,AP2,mAP