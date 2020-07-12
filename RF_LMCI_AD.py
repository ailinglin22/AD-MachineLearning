#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:03:37 2018

@author: XIN
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 18:19:24 2018

@author: XIN
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

@author: XIN
"""

#Features without image features:


import matplotlib.pyplot as plt

from sklearn import svm
from scipy import interp
import numpy as np
import pandas as pd
from itertools import cycle
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
#from imblearn.combine import SMOTEENN
#from imblearn.over_sampling import SMOTE 


#data=pd.read_csv('/Users/XIN/Desktop/NewData/FDG_AB_Merge_WMH_Cognitive_New.csv').fillna(0)
#data=pd.read_csv('/Users/XIN/Documents/XIN/uky/Lin/ADNI/FDG_AB_Merge_WMH_M.csv').fillna(0)
#data=pd.read_csv('/Users/XIN/Documents/XIN/uky/Lin/ADNI/FDG_AB_Merge_WMH_Cognitive_Normalize_All.csv').fillna(0)
#data=pd.read_csv('/Users/XIN/Documents/XIN/uky/Lin/ADNI/FDG_AB_Merge_WMH_Cognitive_Normalize.csv').fillna(0)
#data=pd.read_csv('/Users/XIN/Documents/XIN/uky/Lin/ADNI/Merge_Test.csv')
#data=pd.read_csv('/Users/xinxing/Documents/XIN/Work/DrLin/ADNI/NewData/Merge_all_test.csv').fillna(0)
data=pd.read_csv('/Users/xinxing/Documents/XIN/Work/DrLin/ADNI/NewData/combined_norm_features_final.csv').fillna(0)


#1. All feature
df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost','Entorhinal','Hippocampus','AB_Cingulate','AB_Frontal','AB_Parietal','AB_Temporal','AB_Hippocampus','AB_Precuneus','Ventricles','WholeBrain','WMH','WHITE','GRAY','PTAU']]


#Top 8 features
df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost','AB_Precuneus','PTAU','Entorhinal','Hippocampus','Ventricles']]

#3. AB only
#df=data[['DX','ABCingulate_N','ABFrontal_N','ABParietal_N','ABTemporal_N','ABHippocampus_N','ABPrecuneus_N']]
#df=data[['DX','ABFRONTAL', 'ABCINGULATE', 'ABPARIETAL','ABTEMPORAL','ABPOSTERIORCINGULATE','ABPRECUNEUS', 'ABHIPPOCAMPUS']]
#df=data[['DX','ABFrontal_N','ABParietal_N','ABTemporal_N','ABHippocampus_N','ABPrecuneus_N']]

#4. ABFrontal & ABPRECUNEUS
#df=data[['DX','ABPRECUNEUS', 'ABHIPPOCAMPUS']]

#5. AB wo ABHippo & ABPRECUNEUS
#df=data[['DX','ABFRONTAL', 'ABCINGULATE', 'ABPARIETAL','ABTEMPORAL','ABPOSTERIORCINGULATE']]

#6. FDG only
#df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost']]

#7. Hippo+Entro 
#df=data[['DX','Hippocampus','Entorhinal']]

#8. FDG+Hippo+Entro+ABHippo & ABPRECUNEUS
#df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost','Entorhinal','Hippocampus','ABPRECUNEUS', 'ABHIPPOCAMPUS']]
#df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost','Entorhinal','Hippocampus','ABFrontal_N','ABPrecuneus_N','PTAU']]

#9. FDG+AB
#df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost','AB','ABFRONTAL', 'ABCINGULATE', 'ABPARIETAL','ABTEMPORAL','ABPOSTERIORCINGULATE','ABPRECUNEUS', 'ABHIPPOCAMPUS']]

#10.FDG+ABHippo & ABPRECUNEUS
#df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost','ABHippo_N','ABPrecus_N']]

#11. FDG+Hippo+Entro
#df=data[['DX','FDGAngular','FDGTemporal','FDGCingulumPost','Entorhinal','Hippocampus']]


#12. AB+Hippo+Entor
#df=data[['DX','Entorhinal','Hippocampus','ABFrontal_N','ABParietal_N','ABTemporal_N','ABHippocampus_N','ABPrecuneus_N']]

df_CN=df[df.DX==0] #The number is 226
#df_EMCI=df[df.DX==1] #The number is 521
df_LMCI=df[df.DX==2] #The number is 144
df_AD=df[df.DX==3]
#df_MCI=pd.concat([df_LMCI, df_EMCI])
#df['DX'].value_counts()

#Try the original:
df_sampled=pd.concat([df_LMCI, df_AD])
df_sampled=shuffle(df_sampled)
y=df_sampled.DX
X=df_sampled.drop('DX',axis=1)
X_1=df_sampled.drop('DX',axis=1)
#b=np.mean(importance, axis=0)
#feature_importances = pd.DataFrame(b,index = X_1.columns, columns=['importance']).sort_values('importance',ascending=False)

y = label_binarize(y, classes=[2, 3])

numFeature=X.shape[1]
#X=X.as_matrix()
y=y.ravel()
#normalized standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)



#clf=RandomForestClassifier(n_estimators=100,criterion='entropy',class_weight='balanced')
#clf=svm.SVC(kernel='rbf',C=1000, gamma=0.01,coef0=0.01,probability=True)
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importance=np.zeros((500,numFeature))
res=[]
f1=[]

i = 0
j=0
for i in range(100):
    #clf=RandomForestClassifier(n_estimators=100,criterion='gini',class_weight='balanced')
    clf = GradientBoostingClassifier(n_estimators=100)
    cv = StratifiedKFold(n_splits=5)
    for train, test in cv.split(X, y):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test]) #original
        accuracy=clf.predict(X[test])
        res.append(clf.score(X[test],y[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(X[test])
        f1.append(f1_score(y[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importance[j,:]=clf.feature_importances_
        '''
        #result=result.append(accuracy)
        #print(accuracy)
        #prob_y_0 = [p[1] for p in probas_]
        #print(roc_auc_score(y[test], prob_y_0))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1],drop_intermediate='False')
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        '''
        j += 1
    i += 1
b=np.mean(importance, axis=0)
feature_importances = pd.DataFrame(b,index = X_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print('done')
print(feature_importances)

'''
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Groudline', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
'''