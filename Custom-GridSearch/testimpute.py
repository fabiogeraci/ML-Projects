import numpy as np
from kmodes.kmodes import KModes
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import preprocessing
from mining_tools import my_tools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn import tree

class Imputation_from_Scratch:
    def start_imputation(self,path):

        dataRaw = []
        DataFile = open(path, "r")
        i=0
        while True:
            theline = DataFile.readline()
            theline=theline.replace('\n','')

            if len(theline) == 0:
                break  
            readData = theline.split(",")
            dataRaw.append(readData)

        DataFile.close()

        data = np.array(dataRaw)

        #print(data)
        for i in range(data.shape[1]):
            a,count=np.unique(data[:,i],return_counts=True)
            #print(a,count)
            mode=a[np.argmax(count)] #determining the mode of every feature & replecing it with the missing value
            #print(mode)
            data[data[:,i]=='?',i]=mode
        #print(data)
        label_encoder = LabelEncoder()

        for i in range(data.shape[1]):
            data[:, i] = label_encoder.fit_transform(data[:, i])
            #data[:, i] = fh.fit_transform(test.iloc[:,1])
        data=data.astype(np.float)
        return data



#This program deals with missing data with mode based imputation since all of it's features are categorical
i=Imputation_from_Scratch()
data=i.start_imputation('agaricus-lepiota.data')
print(data.shape)
print(type(data))
y=data[:,0]
y_label=[]
for i in range(len(y)):
    temp=y[0]
    if temp==y[i]:
        y_label.append(1)
    else:
        y_label.append(0)
print(y)

#print(y.shape)
#K-Modes clustering technique invented by Huang in 1997 to cluster categorical features
km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(data[:,1:])
#print(clusters)
print(accuracy_score(y_label,clusters))
mt=my_tools(data[:,1:],y_label,'Mushroom Balanced',2)
#mt.full_throttle()
mt.supervised_only()
mt.call_my_nn()

