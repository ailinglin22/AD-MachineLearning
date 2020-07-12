#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:36:30 2019

@author: xinxing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

import matplotlib.gridspec as gridspec

# Create data

#data=pd.read_csv('/Users/xinxing/Documents/XIN/Work/DrLin/ADNI/NewData/combined_norm_features_final.csv',encoding="mac_greek").fillna(0)
data=pd.read_csv('/Users/xinxing/Google Drive File Stream/My Drive/xin/study/linlab/ADNI/NewData/combined_norm_features_final.csv',encoding='mac_greek').fillna(0)

df=data[['DX','FDG-Angular','Aß-Cingulate','Aß-Frontal','Aß-Temporal','Aß-Precuneus','pTau','Entorhinal','Hippocampus','ADNI_MEM','ADNI_EF']]
df_LMCI=df[df.DX==2]
df_CN=df[df.DX==0]

colors = ("blue",  "magenta")
groups = ("CU", "LMCI")

# Create plot
fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(nrows=2, ncols=4,hspace=0.3)

#FDGAngular
g1_1=df_CN[['FDG-Angular','ADNI_MEM']]
g2_1=df_LMCI[['FDG-Angular','ADNI_MEM']]

g1_1=g1_1.as_matrix()
g2_1=g2_1.as_matrix()

d1_1=(g1_1[:,0],g1_1[:,1])
d2_1=(g2_1[:,0],g2_1[:,1])

group_1=np.concatenate((g1_1,g2_1))

data_1=(d1_1,d2_1)


X_train_1=group_1[:,0]
X_train_1=np.reshape(X_train_1,(-1,1))
y_train_1=group_1[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_1, y_train_1)

X_test_1=np.concatenate((X_train_1[:10],X_train_1[:-10]))
X_test_1=np.reshape(X_test_1,(-1,1))
y_pred_1=regr.predict(X_test_1)

ax1 = fig.add_subplot(gs[1,1])
for data, color, group in zip(data_1, colors, groups):
    x, y = data
    ax1.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax1.set_xlabel('FDGAngular')
#ax1.set_xlim((0,2))
#ax1.set_xticks(np.arange(0,2, 0.5))
from matplotlib.offsetbox import AnchoredText
at = AnchoredText("r=0.3777",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper left',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.plot(X_test_1, y_pred_1, color='black', linewidth=2)
ax1.set_title('FDG-Angular', fontweight='bold', fontsize=25)


#ABPrecuneus_N'
g1_2=df_CN[['Aß-Precuneus','ADNI_MEM']]
g2_2=df_LMCI[['Aß-Precuneus','ADNI_MEM']]

g1_2=g1_2.as_matrix()
g2_2=g2_2.as_matrix()

d1_2=(g1_2[:,0],g1_2[:,1])
d2_2=(g2_2[:,0],g2_2[:,1])

group_2=np.concatenate((g1_2,g2_2))

data_2=(d1_2,d2_2)


X_train_2=group_2[:,0]
X_train_2=np.reshape(X_train_2,(-1,1))
y_train_2=group_2[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_2, y_train_2)

X_test_2=np.concatenate((X_train_2[:10],X_train_2[:-10]))
X_test_2=np.reshape(X_test_2,(-1,1))
y_pred_2=regr.predict(X_test_2)

ax2 = fig.add_subplot(gs[0,1])
for data, color, group in zip(data_2, colors, groups):
    x, y = data
    ax2.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax2.set_xlabel('ABPrecuneus_N')
ax2.set_xlim((0,0.0004))
ax2.set_xticks(np.arange(0,0.0004, 0.0001))
at = AnchoredText("r=-0.4416",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper right',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)
ax2.plot(X_test_2, y_pred_2, color='black', linewidth=2)
ax2.set_title('A'r'$ \beta $''-Precuneus', fontweight='bold', fontsize=25)


#ABFrontal_N
g1_3=df_CN[['Aß-Frontal','ADNI_MEM']]
g2_3=df_LMCI[['Aß-Frontal','ADNI_MEM']]

g1_3=g1_3.as_matrix()
g2_3=g2_3.as_matrix()

d1_3=(g1_3[:,0],g1_3[:,1])
d2_3=(g2_3[:,0],g2_3[:,1])

group_3=np.concatenate((g1_3,g2_3))

data_3=(d1_3,d2_3)


X_train_3=group_3[:,0]
X_train_3=np.reshape(X_train_3,(-1,1))
y_train_3=group_3[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_3, y_train_3)

X_test_3=np.concatenate((X_train_3[:10],X_train_3[:-10]))
X_test_3=np.reshape(X_test_3,(-1,1))
y_pred_3=regr.predict(X_test_3)

ax3 = fig.add_subplot(gs[0,3])
for data, color, group in zip(data_3, colors, groups):
    x, y = data
    ax3.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax3.set_xlabel('ABFrontal_N')
ax3.set_xlim((0,0.000025))
ax3.set_xticks(np.arange(0,0.000025, 0.0000125))
at = AnchoredText("r=-0.4244",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper right',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at)
ax3.plot(X_test_3, y_pred_3, color='black', linewidth=2)
ax3.set_title('A'r'$ \beta $''-Frontal', fontweight='bold', fontsize=25)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 20})


#ABTemporal_N
g1_4=df_CN[['Aß-Temporal','ADNI_MEM']]
g2_4=df_LMCI[['Aß-Temporal','ADNI_MEM']]

g1_4=g1_4.as_matrix()
g2_4=g2_4.as_matrix()

d1_4=(g1_4[:,0],g1_4[:,1])
d2_4=(g2_4[:,0],g2_4[:,1])

group_4=np.concatenate((g1_4,g2_4))

data_4=(d1_4,d2_4)


X_train_4=group_4[:,0]
X_train_4=np.reshape(X_train_4,(-1,1))
y_train_4=group_4[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_4, y_train_4)

X_test_4=np.concatenate((X_train_4[:10],X_train_4[:-10]))
X_test_4=np.reshape(X_test_4,(-1,1))
y_pred_4=regr.predict(X_test_4)

ax4 = fig.add_subplot(gs[0,0])
for data, color, group in zip(data_4, colors, groups):
    x, y = data
    ax4.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax4.set_xlabel('ABTemporal_N')
ax4.set_xlim((0,0.00006))
ax4.set_xticks(np.arange(0,0.00006, 0.00003))
at = AnchoredText("r=-0.4670",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper right',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax4.add_artist(at)
ax4.plot(X_test_4, y_pred_4, color='black', linewidth=2)
ax4.set_title('A'r'$ \beta $''-Temporal', fontweight='bold', fontsize=25)

#Hippocampus
g1_5=df_CN[['Hippocampus','ADNI_MEM']]
g2_5=df_LMCI[['Hippocampus','ADNI_MEM']]

g1_5=g1_5.as_matrix()
g2_5=g2_5.as_matrix()

d1_5=(g1_5[:,0],g1_5[:,1])
d2_5=(g2_5[:,0],g2_5[:,1])

group_5=np.concatenate((g1_5,g2_5))

data_5=(d1_5,d2_5)

X_train_5=group_5[:,0]
X_train_5=np.reshape(X_train_5,(-1,1))
y_train_5=group_5[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_5, y_train_5)

X_test_5=np.concatenate((X_train_5[:10],X_train_5[:-10]))
X_test_5=np.reshape(X_test_5,(-1,1))
y_pred_5=regr.predict(X_test_5)

ax5 = fig.add_subplot(gs[0,2])
for data, color, group in zip(data_5, colors, groups):
    x, y = data
    ax5.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax5.set_xlabel('Hippocampus')
#ax3.set_xlim((0,2))
#ax3.set_xticks(np.arange(0,2, 0.5))
at = AnchoredText("r=0.4414",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper left',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax5.add_artist(at)
ax5.plot(X_test_5, y_pred_5, color='black', linewidth=2)
ax5.set_title('Hippocampus', fontweight='bold', fontsize=25)


#Entorhinal
g1_6=df_CN[['Entorhinal','ADNI_MEM']]
g2_6=df_LMCI[['Entorhinal','ADNI_MEM']]

g1_6=g1_6.as_matrix()
g2_6=g2_6.as_matrix()

d1_6=(g1_6[:,0],g1_6[:,1])
d2_6=(g2_6[:,0],g2_6[:,1])

group_6=np.concatenate((g1_6,g2_6))

data_6=(d1_6,d2_6)

X_train_6=group_6[:,0]
X_train_6=np.reshape(X_train_6,(-1,1))
y_train_6=group_6[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_6, y_train_6)

X_test_6=np.concatenate((X_train_6[:10],X_train_6[:-10]))
X_test_6=np.reshape(X_test_6,(-1,1))
y_pred_6=regr.predict(X_test_6)

ax6 = fig.add_subplot(gs[1,3])
for data, color, group in zip(data_6, colors, groups):
    x, y = data
    ax6.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax6.set_xlabel('Entorhinal')
#ax3.set_xlim((0,2))
#ax3.set_xticks(np.arange(0,2, 0.5))
at = AnchoredText("r=0.3143",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper left',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax6.add_artist(at)
ax6.plot(X_test_6, y_pred_6, color='black', linewidth=2)
ax6.set_title('Entorhinal Cortex', fontweight='bold', fontsize=25)


#ABCingulare
g1_7=df_CN[['Aß-Cingulate','ADNI_MEM']]
g2_7=df_LMCI[['Aß-Cingulate','ADNI_MEM']]

g1_7=g1_7.as_matrix()
g2_7=g2_7.as_matrix()

d1_7=(g1_7[:,0],g1_7[:,1])
d2_7=(g2_7[:,0],g2_7[:,1])

group_7=np.concatenate((g1_7,g2_7))

data_7=(d1_7,d2_7)

X_train_7=group_7[:,0]
X_train_7=np.reshape(X_train_7,(-1,1))
y_train_7=group_7[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_7, y_train_7)

X_test_7=np.concatenate((X_train_7[:10],X_train_7[:-10]))
X_test_7=np.reshape(X_test_7,(-1,1))
y_pred_7=regr.predict(X_test_7)

ax7 = fig.add_subplot(gs[1,2])
for data, color, group in zip(data_7, colors, groups):
    x, y = data
    ax7.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax7.set_xlabel('ABHippocampus_N')
ax7.set_xlim((0, 0.00015))
ax7.set_xticks(np.arange(0, 0.00015, 0.00005))
at = AnchoredText("r=-0.3587",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper right',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax7.add_artist(at)
ax7.plot(X_test_7, y_pred_7, color='black', linewidth=2)
ax7.set_title('A'r'$ \beta $''-Cingulate', fontweight='bold', fontsize=25)


#PTau
g1_8=df_CN[['pTau','ADNI_MEM']]
g2_8=df_LMCI[['pTau','ADNI_MEM']]

g1_8=g1_8.as_matrix()
g2_8=g2_8.as_matrix()

d1_8=(g1_8[:,0],g1_8[:,1])
d2_8=(g2_8[:,0],g2_8[:,1])

group_8=np.concatenate((g1_8,g2_8))

data_8=(d1_8,d2_8)

X_train_8=group_8[:,0]
X_train_8=np.reshape(X_train_8,(-1,1))
y_train_8=group_8[:,1]

#y_train=np.reshape(y_train,(-1,1))
regr = linear_model.LinearRegression()
regr.fit(X_train_8, y_train_8)

X_test_8=np.concatenate((X_train_8[:10],X_train_8[:-10]))
X_test_8=np.reshape(X_test_8,(-1,1))
y_pred_8=regr.predict(X_test_8)

ax8 = fig.add_subplot(gs[1,0])
for data, color, group in zip(data_8, colors, groups):
    x, y = data
    ax8.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=40, label=group)
#ax8.set_xlabel('PTAU')
#ax3.set_xlim((0,2))
#ax3.set_xticks(np.arange(0,2, 0.5))
at = AnchoredText("r=-0.4243",
                  prop=dict(size=18,fontweight="bold"), frameon=False,
                  loc='upper right',
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax8.add_artist(at)
ax8.plot(X_test_8, y_pred_8, color='black', linewidth=2)
ax8.set_title('pTau', fontweight='bold', fontsize=25)


fig.text(0.08, 0.5, 'Composite Memory \n Score', ha='center', va='center', rotation='vertical',fontsize=35, fontweight='bold')
#fig.text(0.1, 0.9, 'a.', ha='center', va='center', fontsize=8, fontweight='bold')


plt.show()
fig.savefig("/Users/xinxing/Desktop/CNvsLMCI_MEM.pdf")