from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from random import randrange
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os,stat
import pydotplus
import matplotlib.pyplot as plt


class validation_Accuracy:
    def __init__(self):
        self.precision=[]
        self.recall=[]
        self.f1=[]
        self.accuracy=[]
        self.sensitivity=[]
        self.specificity=[]

    def kfold_crossvalidation(self,data,kfold):
        kfold_data=list()
        fold_size=len(data)/kfold
        for i in range(kfold):
            r_fold=list()
            while(len(r_fold)<fold_size):
                r_fold.append(data[randrange(len(data))])
            kfold_data.append(r_fold)
        return kfold_data


    def calculate_accuracy(self,y_test,predictions):
        correct_result=0
        for i in range(len(y_test)):
            if y_test[i]==predictions[i]:
                correct_result+=1
        acc=correct_result/float(len(y_test))*100
        conf=confusion_matrix(y_test,predictions)
        print(conf)
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



data=pd.read_csv('model_df.csv')
y=data.iloc[:,-1].values.astype(np.float)
x=data.iloc[:,:-1]
label_encoder = LabelEncoder()
for i in range(5):
    x.iloc[:, i] = label_encoder.fit_transform(x.iloc[:, i])
x.iloc[:, 7] = label_encoder.fit_transform(x.iloc[:, 7])
#print(x)
sc=StandardScaler()
x_data=sc.fit_transform(x)
#print(x_data)
# Checking Class Imbalance
a = pd.value_counts(y, sort=False)
print(a)
y=y.reshape((-1,1))
p_data=np.append(x_data,y,axis=1)
ob=validation_Accuracy()
k_data=ob.kfold_crossvalidation(p_data,5)
# trainset=list()
#testset=list()
#print(k_data[0])
for fold in k_data:
    #print(k_data[i])
    i=0
    trainset=list(k_data)
    trainset.pop(i)
    trainset=sum(trainset,[])
    trainset=np.array(trainset)
    testset=list(fold)
    testset=np.array(testset)
    x_train=np.array(trainset[:,:-1])
    y_train=np.array(trainset[:,-1])
    x_test=np.array(testset[:,:-1])
    y_test=np.array(testset[:,-1])

    #x_train, x_test, y_train, y_test = train_test_split(temp, [row[-1] for row in k_data[i]])
    #y_test=np.array(y_test).reshape((-1,1))
    clf=RandomForestClassifier(max_depth=5)
    #clf=tree.DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    feature_importances = pd.DataFrame(clf.feature_importances_,index = x.columns,columns=['importance']).sort_values('importance',ascending=False)
    print(feature_importances)
    y_pred=clf.predict(x_test)
    
    ob.calculate_accuracy(y_test,y_pred)
    i+=1

estimator = clf.estimators_[2]
feature_name=x.columns
print(feature_name)
dot_data = tree.export_graphviz(estimator, out_file="decision_tree.dot",feature_names =feature_name,filled=True,class_names =['checkout','No Checkout'], rounded=True,proportion = False,precision = 2)

from subprocess import check_call
check_call(['dot', '-Tpng', 'decision_tree.dot', '-o', 'tree.png', '-Gdpi=600'])


print('Accuracy',ob.accuracy)
print('Precision',ob.precision)
print('Recall',ob.recall)
print('F1 score',ob.f1)
print('Sensitivity', ob.sensitivity)
print('Specificity' , ob.specificity)

# sorting values
feature_importances.sort_values('importance',inplace=True)
#print(feature_importances)

#plt.figure(figsize=(10,7))
ax=feature_importances.plot.barh(y='importance',rot=0, align='edge')
ax.set_xlabel("Importance",fontsize=12)
ax.set_ylabel("Features",fontsize=12)
plt.savefig('feature_importance.jpg')

plt.show()


