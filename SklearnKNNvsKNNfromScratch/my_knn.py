#Test case of KNN on breast-cancer dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from random import randrange
import random
import pandas as pd
import numpy as np
import math
import time
#This class is designed to calculate accuracy,precision,recall & all the other performance metrics for both algorithms
class Accuracy:
    def __init__(self,n_neighbors):
        self.precision=[]
        self.recall=[]
        self.f1=[]
        self.accuracy=[]
        self.sensitivity=[]
        self.specificity=[]
        #self.identify=[]
        #self.neighbors=[]
        self.neighbors=n_neighbors
    def calculate_accuracy(self,y_test,predictions,flag):
        correct_result=0
        for i in range(len(y_test)):
            if y_test[i]==predictions[i]:
                correct_result+=1
        acc=correct_result/float(len(y_test))*100
        #print(predictions)
        conf=confusion_matrix(y_test,predictions)
        #print(conf)
        tn=conf[0,0]
        tp=conf[1,1]
        fp=conf[0,1]
        fn=conf[1,0]
        p=tp/(tp+fp)
        r=tp/(tp+fn)
        f=2*(1/(1/p + 1/r))
        sen=tp/(fn+tp)
        spe=fp/(fp+tn)
        self.accuracy.append(acc)
        self.precision.append(p)
        self.recall.append(r)
        self.f1.append(f)
        self.sensitivity.append(sen)
        self.specificity.append(spe)
        if flag==0:
            self.identify='KNN FROM SCRATCH'
        else:
            self.identify='KNN SKLEARN'


class KNN_from_Scratch:

        

    def kfold_crossvalidation(self,data,kfold): #Mathod for k fold cross validation, here this program is using 5 fold
        kfold_data=list()
        fold_size=len(data)/kfold #fold size self explanatory
        for i in range(kfold):
            r_fold=list()
            while(len(r_fold)<fold_size): #randomly choosing the rows till it reaches the fold size
                r_fold.append(data[randrange(len(data))])
            kfold_data.append(r_fold)
        return kfold_data

    def calculate_distance(self,p1,p2): #calculating euclidean distance
        dist=0
        for i in range(len(p1)-1):
            dist = dist + pow((p1[i] - p2[i]),2)
        return math.sqrt(dist) 
    def find_neighbors(self,train,test,k):
        dist=list()
        for train_row in train: #for each training row it is calculating distance between train & for the parameterised test row
            d=self.calculate_distance(test,train_row)
            dist.append((train_row,d))
            dist.sort(key=lambda tup: tup[1]) #making a tuple of train row & it's distance
        k_neighbors=list()
        for i in range(k):
            k_neighbors.append(dist[i][0]) #selecting only nearest neighbors based on k value i.e only the row not the distance
        return k_neighbors
    def predict_labels(self,train,test,k):
        knn=self.find_neighbors(train,test,k)
        #print(knn)
        knn_votes=[row[-1] for row in knn] #labels of all nearest neighbors
        #print(knn_votes)
        knn_pred=max(set(knn_votes), key=knn_votes.count) #maximum number of votes or most neighbors that have same label
        return knn_pred

data=pd.read_csv('breast-cancer.data')
data=data.iloc[1:,:]
data=data.iloc[random.sample(range(1,650),300),:] #choosing 300 rows randomly otherwise it's taking a looooong time to execute

y=data.iloc[:,-1].values.astype(np.float)
y=y.reshape((-1,1)) #1d to 2d array this is required for np.append

sc=StandardScaler() #scaling the data
x_data=sc.fit_transform(data.iloc[:,:-1])
data=np.append(x_data,y,axis=1)

#print(k_data)
accuracy=list()
precision=list()
recall=list()
f1=list()
sensitivity=list()
specificity=list()
identify=list()
neighbors=list()
processing_time=list()
#The program is checking performance metric for both algorithms starting from 1 to 50
number_of_neighbors=list(range(1,51))
for n in range(len(number_of_neighbors)):
    processing_time_sklearn=list()
    processing_time_scratch=list()
    ob=KNN_from_Scratch() #calling the object
    ob1=Accuracy(number_of_neighbors[n]) 
    sklearnob=Accuracy(number_of_neighbors[n])
    k_data=ob.kfold_crossvalidation(data,5) #ob is just the object of KNN from scratch but it executes the cross validation method which 
    for fold in k_data: #is used for both sklearn & scratch class as both the class must get the same data for each validation
        i=0
        trainset=list(k_data) #creating training & testset from the folds
        trainset.pop(i)
        trainset=sum(trainset,[])
        trainset=np.array(trainset)
        testset=list(fold)
        testset=np.array(testset)
        x_train=np.array(trainset[:,:-1])
        y_train=np.array(trainset[:,-1])
        x_test=np.array(testset[:,:-1])
        y_test=np.array(testset[:,-1])
        #print(y_train)

        predictions=list()
        #print(np.array(x_test).shape)
        y_test=np.array(y_test).reshape((-1,1)) #1d to 2d array this was used as the class KNN from scratch designed earlier
        sklearn_start=time.clock()
        clf=KNeighborsClassifier(n_neighbors=number_of_neighbors[n])
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test) 
        #print('y_pred',y_pred)
        sklearn_final=time.clock()
        processing_time_sklearn.append(sklearn_final-sklearn_start) #calculating the processing time
        #sklearn_accuracy.append(accuracy_score(y_test,y_pred))
        sklearnob.calculate_accuracy(y_test,y_pred,1) #same accuracy class is used for evaluation metric
        #print(np.array(y_test).shape)
        data_test=np.append(x_test,y_test,axis=1)
        y_train=np.array(y_train).reshape((-1,1))
        data_train=np.append(x_train,y_train,axis=1)
        knn_scratch_start=time.clock()
        for test in data_test:
            pred=ob.predict_labels(data_train,test,number_of_neighbors[i])
            predictions.append(pred)
        #print('Accuracy Score',accuracy_score(y_test,predictions))
        knn_scratch_final=time.clock()

        processing_time_scratch.append(knn_scratch_final-knn_scratch_start) #calculating time
        ob1.calculate_accuracy(y_test,predictions,0)
    accuracy.append(sum(ob1.accuracy)/len(ob1.accuracy)) #adding the average scores into the list for all k numnber of folds
    accuracy.append(sum(sklearnob.accuracy)/len(sklearnob.accuracy))
    identify.append(ob1.identify)
    identify.append(sklearnob.identify)
    neighbors.append(ob1.neighbors)
    neighbors.append(sklearnob.neighbors)
    precision.append(sum(ob1.precision)/len(ob1.precision))
    precision.append(sum(sklearnob.precision)/len(sklearnob.precision))
    recall.append(sum(ob1.recall)/len(ob1.recall))
    recall.append(sum(sklearnob.recall)/len(sklearnob.recall))
    f1.append(sum(ob1.f1)/len(ob1.f1))
    f1.append(sum(sklearnob.f1)/len(sklearnob.f1))
    sensitivity.append(sum(ob1.sensitivity)/len(ob1.sensitivity))
    sensitivity.append(sum(sklearnob.sensitivity)/len(sklearnob.sensitivity))
    specificity.append(sum(ob1.specificity)/len(ob1.specificity))
    specificity.append(sum(sklearnob.specificity)/len(sklearnob.specificity))
    processing_time.append(sum(processing_time_scratch)/len(processing_time_scratch))
    processing_time.append(sum(processing_time_sklearn)/len(processing_time_sklearn))
    
#print()
#print(accuracy)
list_for_export=[identify,neighbors,precision,recall,f1,sensitivity,specificity,processing_time,accuracy] #making the entire list
list_for_export=np.array(list_for_export).transpose()
#print(list_for_export)
export_data=pd.DataFrame(data=list_for_export[0:,0:],columns=['Identifier','Number of Neighbors','Precision','Recall','F1 Score','Sensitivity','Specificity','Processing Time','Accuracy'])
export_data.to_csv('KNN_Export.csv') #exporting into csv file
