"""Confusion matrices and explicit benchmark aggregation."""
import numpy as np

def confusion(prediction,label,classes):
    valid=label!=255
    if prediction.shape!=label.shape:raise ValueError('Prediction/label shape mismatch')
    if np.any((prediction[valid]<0)|(prediction[valid]>=classes)):raise ValueError('Prediction class outside vocabulary')
    return np.bincount(label[valid].astype(np.int64)*classes+prediction[valid],minlength=classes**2).reshape(classes,classes)

def summarize(cm,dataset):
    cm=np.asarray(cm,dtype=np.int64);inter=np.diag(cm);union=cm.sum(0)+cm.sum(1)-inter
    iou=np.divide(inter,union,out=np.full(len(cm),np.nan),where=union>0)*100
    selected=[1] if dataset=='inria' else list(range(1,7)) if dataset=='deepglobe' else list(range(8))
    return {'confusion_matrix':cm.tolist(),'class_iou_percent':[None if np.isnan(x) else float(x) for x in iou],
            'primary_percent':float(np.nanmean(iou[selected])),'selected_classes':selected}
