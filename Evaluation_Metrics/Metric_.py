import numpy as np
from sklearn.metrics import confusion_matrix
def mean_iou(result,Target):
    predicted=np.reshape(result[:,:,:,:],(result.shape[0]*896*896*2,1))
    predicted=predicted.astype(int)
    Target=np.reshape(Target[:,:,:,:],(result.shape[0]*896*896*2,1))
    target=Target.astype(int)
    tn, fp, fn, tp=confusion_matrix(target, predicted).ravel()
    print('TP = ',tp)
    print('FP = ',fp)
    print('FN = ',fn)
    print('TN = ',tn)
    iou=tp/(tp+fn+fp)
    Sensitivity=tp/(tp+fn)
    precision=tp/(tp+fp)
    recal=tp/(tp+fn)
    F1=(2*precision*recal)/(precision+recal)
    print("F1_Score is:  ",F1)
    print("Sensitivity  is:  ",Sensitivity)
    return print("Mean IOU is : " ,iou)
def Crop_iou(result,Target):
    predicted_crop=np.reshape(result[:,:,:,0],(result.shape[0]*896*896*1,1))
    predicted_crop=predicted_crop.astype(int)
    Target=np.reshape(Target[:,:,:,0],(result.shape[0]*896*896*1,1))
    target=Target.astype(int)
    tn, fp, fn, tp=confusion_matrix(target, predicted_crop).ravel()
    iou_crop=tp/(tp+fn+fp)
    print("Crop IOU is : " ,iou_crop)   
def Weed_iou(result,Target):
    predicted_weed=np.reshape(result[:,:, :,1],(result.shape[0]*896*896*1,1))
    predicted_weed=predicted_weed.astype(int)
    Target=np.reshape(Target[:,:,:,1],(result.shape[0]*896*896*1,1))
    target=Target.astype(int)
    tn, fp, fn, tp=confusion_matrix(target, predicted_weed).ravel()
    iou_weed=tp/(tp+fn+fp)
    print("Weed IOU is : " ,iou_weed)

            