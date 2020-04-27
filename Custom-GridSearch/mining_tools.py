
#tools
import numpy as np
import pandas as pd
from matplotlib import cm
import mglearn
import os
import shutil
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN



import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from testnn_new import MyNN_From_Scratch
from KMeans import kmeansByHand
from logisticregression import log_regression_scratch_cv


#PCA from scratch
class PCA_From_Scratch:
    def fit_transform(self,x):
    # calculate the mean of each column
        Mean = mean(x.T, axis=1)
        # center columns by subtracting column means
        Center = x - Mean
        # calculate covariance matrix of centered matrix
        V = cov(Center.T)
        # eigendecomposition of covariance matrix
        values, vectors = eig(V)
        # project data
        components = vectors.T.dot(Center.T)
        components=components.T
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(values[i]), components[:,i]) for i in range(len(values))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()
        ordered_comp=[item[1] for item in eig_pairs]
        comp=np.asarray(ordered_comp)
        comp=comp.T
        return comp

#This class works as a toolset which is designed in a modular approach so that it can be modified or debug with very little effort
#Also it reduces the lines of code used several times within a program which can be very hard to replace in case of future
#modifications. This Structure is very easy to interpret as object oriented principles like abstraction is implemented in a class
#object architecture in this project

class my_tools:
    #variable initialisation
    def __init__(self,x,y,path,k):
        
        self.x=x
        self.y=y
        self.path='Plots '+path
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.mkdir('Plots '+path)
        
        #print('value of y',self.y)
        self.count=0
        self.k_clusters=k
        self.scalar=[]
        self.precision=[]
        self.recall=[]
        self.f1=[]
        self.r_scalar=[]
        self.r_preprocess=[]
        self.r_cluster=[]
        self.r_classifier=[]
        self.accuracy=[]
        self.sensitivity=[]
        self.specificity=[]
        self.scalar_label=[]
        self.scalar.append(MinMaxScaler())
        self.scalar_label.append("MinMax Scaler")
        self.scalar.append(StandardScaler())
        self.scalar_label.append("Standard Scaler")
        self.scalar.append(RobustScaler())
        self.scalar_label.append("Robust Scaler")
        self.scalar.append(Normalizer())
        self.scalar_label.append("Normal Scaler")
        #scaled_data=[]
        self.preprocess_algo=[]
        self.preprocess_algo.append(PCA(n_components=0.95, svd_solver='full'))
        self.preprocess_algo.append(PCA_From_Scratch())
        self.preprocess_algo.append(NMF())
        self.two_components_algo=[]
        self.two_components_algo.append(PCA(n_components=2))
        self.two_components_algo.append(PCA_From_Scratch())
        self.two_components_algo.append(NMF(n_components=2))
        self.two_components_label=[]
        self.two_components_label.append("PCA ")
        self.two_components_label.append("PCA From Scratch")
        self.two_components_label.append("NMF ")

        self.supervised_algo=[]
        self.supervised_algo.append(KNeighborsClassifier(n_neighbors=3))
        self.supervised_algo.append(LogisticRegression())
        self.supervised_algo.append(tree.DecisionTreeClassifier())
        #self.supervised_algo.append(log_regression_scratch_cv())
        self.supervised_label=[]
        self.supervised_label.append("K Nearest Neighbors: ")
        self.supervised_label.append("Logistic Regression: ")
        self.supervised_label.append("Decision Tree:  ")
        #self.supervised_label.append("Logistic Regression Byhand:  ")

        self.clustering_algo=[]
        self.clustering_algo.append(KMeans(n_clusters=self.k_clusters))
        self.clustering_algo.append(DBSCAN())
        self.clustering_algo.append(kmeansByHand(self.k_clusters,self.path))
        self.clustering_label=[]
        self.clustering_label.append("K Means Algorithm")
        self.clustering_label.append("DBSCAN ")
        self.clustering_label.append("K Means ByHand")
        self.clustering_binary_classifier_algo=[]
        self.clustering_binary_classifier_algo.append(KMeans(n_clusters=2))
        self.clustering_binary_classifier_algo.append(kmeansByHand(2,self.path))
        self.clustering_binary_classifier_label=[]
        self.clustering_binary_classifier_label.append("K Means Algorithm")
        self.clustering_binary_classifier_label.append("K Means ByHand")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=0)

    #One generic method to apply all machine learning classifiers & therefore the classifier that were implemented from scratch
    #were developed in a way so that they also obey the same architecture to generalize all classifiers with one method.
    def my_Classifiers(self,classifier,x_train,y_train,x_test,y_test):
        clf=classifier
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test)
        conf=confusion_matrix(y_test,y_pred)
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
        self.precision.append(p)
        self.recall.append(r)
        self.f1.append(f)
        self.sensitivity.append(sen)
        self.specificity.append(spe)


        score=clf.score(x_test,y_test)
        self.f1_score=f1_score(y_test,y_pred)
        print(classification_report(y_test,y_pred))
        return score

    #One general method to fit & predict all scalars
    def my_Scaler(self,x,scalar_name):
        scaler=scalar_name
        x_transformed=scaler.fit_transform(x)
        return x_transformed

    #This method provides very useful insights about how many PCA components should be considered for feature extraction
    #This method also creates the curve of explained variance for number of components in PCA
    def check_PCA_Components(self,x,s):
        c_pca=PCA().fit(x)
        plt.figure(figsize=(10,7))
        var=np.cumsum(c_pca.explained_variance_ratio_)
        plt.plot(var)
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')
        plt.title("Explained variance of PCA for "+s)
        self.count=self.count+1
        #plt.colorbar(sc)
        plt.show(block=False)
        plt.pause(2)
        plt.savefig(self.path+'//test'+str(self.count)+'.jpg')
        plt.close()
        return var

    #Backup function
    def my_PCA(self,x):
        pca=PCA(n_components=0.99, svd_solver='full')
        x_transformed=pca.fit_transform(x)
        #print(x_transformed)
        return x_transformed
    #One generic function to deal with PCA & NMF 
    def my_preprocessing(self,x,algo):
        algorithm=algo
        
        try:
            x_transformed=algorithm.fit_transform(x)

        except:
            x_transformed=np.array([1])
        #print(x_transformed)
        return x_transformed
    #One generic function to apply all clustering techniques with one method
    def my_model_clusters(self,x,y,algo,algo_label,flag):
        model=algo
        #model.fit(x)
        prediction= model.fit_predict(x)
        if flag==1:  #flag value for clusturing as classification

            print("Prediction Accuracy: ",algo_label,accuracy_score(y,prediction))
        elif flag==2:   #flag value for clustering as feature engineering
            label=model.labels_.reshape((-1,1))
            output=np.append(x,label,axis=1)
            #print(output)
            return output
        else:   #dummy doing nothing 
            x['Kmeans Clusters']=model.labels_
            print(x)
            #print(model.labels_.shape)
            return x


    #backup function to test
    def my_NMF(self,x):
        nmf=NMF()
        x_transformed=nmf.fit_transform(x)
        return x_transformed
    #removed from the program architecture since it's taking a long time to proces 
    def tsne_Manifold(self,x):
        tsne=TSNE()
        x_transformed=tsne.fit_transform(x)
        return x_transformed

    #creates a scatter plot when PCA or NMF components need to produce a scatter plot
    def scatter_Plot(self,x,y,xlab,ylab,title):
        plt.figure(figsize=(10,7))
        sc=plt.scatter(x,y,s=7)
        #plt.rcParams["figure.figsize"] = (10,7)
        #,cmap=plt.cm.get_cmap('RdYlBu'))
        
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        self.count=self.count+1
        #plt.colorbar(sc)
        plt.show(block=False)
        plt.pause(2)
        plt.savefig(self.path+'//test'+str(self.count)+'.jpg')
        #plt.pause(3)
        
        #plt.pause(3)
        plt.close()
    
    #This method only applies classifiers directly to the data, this is a one of ways to start with classification without any preprocessing 
    #to compare between the performance of the models.
    def supervised_only(self):
        for k in range(len(self.supervised_algo)):
            score=self.my_Classifiers(self.supervised_algo[k],self.x_train,self.y_train,self.x_test,self.y_test)
            print("Prediction Accuracy: ",self.supervised_label[k],score)
            self.r_classifier.append(self.supervised_label[k])
            self.r_scalar.append('NA')
            self.r_cluster.append('NA')
            self.r_preprocess.append('NA')
    #Here 4 different scalars has been used & scaled data has been used to train & test the classifiers.
    def scaled_and_supervised(self):
        #Data scaled & then applied on classifiers
        for i in range(len(self.scalar)):
            scaled_data=self.my_Scaler(self.x,self.scalar[i])
            #var=check_PCA_Components(scaled_data,scalar_label[i])
            x_train, x_test, y_train, y_test = train_test_split(scaled_data, self.y, random_state=0)
            for k in range(len(self.supervised_algo)):
                score=self.my_Classifiers(self.supervised_algo[k],x_train,y_train,x_test,y_test)
                print("Prediction Accuracy: ",self.scalar_label[i],self.supervised_label[k],score)
                self.r_classifier.append(self.supervised_label[k])
                self.r_scalar.append(self.scalar_label[i])
                self.r_cluster.append('NA')
                self.r_preprocess.append('NA')

    #Here data is first scaled then for each scalar PCA or NMF has been implemented.
    #As a variation PCA 2 components has been considered since that's the best way to visualise the data.
    #At the same time it also deals with loss of information so multi components were taken into considered where,
    #number of components have a sum of 95% explained variance of the entire dataset.
    #Similarly  NMF is used for 2 components & for all components since NMF doen't have an explained variance & we don't want to
    #lost information on additional features. These combination along with scaling is then forwarded to the supervised machine learning
    #classifiers.
    def scaled_preprocessed_supervised(self):
        # Data Scaled & preprocessed then applied on classifiers
        for i in range(len(self.scalar)):
            scaled_data=self.my_Scaler(self.x,self.scalar[i])
            var=self.check_PCA_Components(scaled_data,self.scalar_label[i])
            #print("Explained Variance",scalar_label[i],var)
            for j in range(len(self.preprocess_algo)):
                #print(i,j)
                processed_data=self.my_preprocessing(scaled_data,self.preprocess_algo[j])
                #print("Multi Component",processed_data)
                
                processed_two_components=self.my_preprocessing(scaled_data,self.two_components_algo[j])
                #print("Two Component",processed_two_components)
                
                shape=processed_two_components.shape
                #print("Length",len(list(processed_two_components)))
                if len(list(processed_two_components))>1:
                    self.scatter_Plot(processed_two_components[:,0],processed_two_components[:,1],"Component 1","Component 2",self.two_components_label[j]+"Scatter Plot"+self.scalar_label[i])
                    x_train, x_test, y_train, y_test = train_test_split(processed_two_components, self.y, random_state=0)
                    x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(processed_data, self.y, random_state=0)
                else:
                    #since NMF does not work with negative values
                    print("NMF can not be applied to non negative matrix of the corresponding scaled data",self.scalar_label[i])
                    continue
                for k in range(len(self.supervised_algo)):
                    score=self.my_Classifiers(self.supervised_algo[k],x_train,y_train,x_test,y_test)
                    print("Prediction Accuracy: ",self.scalar_label[i],self.two_components_label[j]+ "Two Components ",self.supervised_label[k],score)
                    #variables that stores information which will produce our autogenerated comparison report of several models.
                    self.r_classifier.append(self.supervised_label[k])
                    self.r_scalar.append(self.scalar_label[i])
                    self.r_cluster.append('NA')
                    self.r_preprocess.append(self.two_components_label[j])
                    score_m=self.my_Classifiers(self.supervised_algo[k],x_m_train,y_m_train,x_m_test,y_m_test)
                    print("Prediction Accuracy: ",self.scalar_label[i],self.two_components_label[j]+ "Multi Components",self.supervised_label[k],score_m)
                    self.r_classifier.append(self.supervised_label[k])
                    self.r_scalar.append(self.scalar_label[i])
                    self.r_cluster.append('NA')
                    self.r_preprocess.append(self.two_components_label[j])
    #It is self explanatory, here the predicted labels are used to measure performance of the algorithm with true value of labels
    def clustering_as_classification(self):
        #Clusturing Technique on a classification problem
        for m in range(len(self.clustering_binary_classifier_algo)):
            self.my_model_clusters(self.x,self.y,self.clustering_binary_classifier_algo[m],self.clustering_label[m],1)
            self.r_classifier.append('NA')
            self.r_scalar.append('NA')
            self.r_cluster.append(self.clustering_label[m])
            self.r_preprocess.append('NA')
    #This technique very useful as clustering can be used as another feature accross the other features to accurately predict output
    #Here KMeans, KMeans ByHand & DBSCAN were applied & then the output is forwarded to supervised machine learning algorithms.
    def clustering_and_supervised(self):
        #Feature engineering using clustering & then data applied on classifiers
        for m in range(len(self.clustering_algo)):
            clustered_data=self.my_model_clusters(self.x,self.y,self.clustering_algo[m],self.clustering_label[m],2)
            x_train, x_test, y_train, y_test = train_test_split(clustered_data, self.y, random_state=0)
            for k in range(len(self.supervised_algo)):
                score=self.my_Classifiers(self.supervised_algo[k],x_train,y_train,x_test,y_test)
                print("Prediction Accuracy: ",self.clustering_label[m],self.supervised_label[k],score)
                self.r_classifier.append(self.supervised_label[k])
                self.r_scalar.append('NA')
                self.r_cluster.append(self.clustering_label[m])
                self.r_preprocess.append('NA')

    #This is a comprehensive process to apply every stages involved in building a model for classification problems.
    #This method utilizes every possible combination of these four stages to determine the best model for the dataset along with
    #models that were developed before
    def scaled_pre_clus_supervised(self):
        #Data scaled, preprocessed, clustered to add feature & then applied on classifiers
        new_x=self.x.iloc[:,:-1]

        for i in range(len(self.scalar)):
            dsp_scaled_data=self.my_Scaler(new_x,self.scalar[i])
            dsp_var=self.check_PCA_Components(dsp_scaled_data,self.scalar_label[i])
            #print("Explained Variance",scalar_label[i],var)
            for j in range(len(self.preprocess_algo)):
                #print(i,j)
                dsp_processed_data=self.my_preprocessing(dsp_scaled_data,self.preprocess_algo[j])
                #print("Multi Component",processed_data)
                
                dsp_processed_two_components=self.my_preprocessing(dsp_scaled_data,self.two_components_algo[j])
                #print("Two Component",processed_two_components)
                
                #shape=processed_two_components.shape
                #print("Length",len(list(processed_two_components)))
                if len(list(dsp_processed_two_components))>1:
                    self.scatter_Plot(dsp_processed_two_components[:,0],dsp_processed_two_components[:,1],"Component 1","Component 2",self.two_components_label[j]+"Scatter Plot"+self.scalar_label[i])
                    for m in range(len(self.clustering_algo)):
                        clustered_tc_data=self.my_model_clusters(dsp_processed_two_components,self.y,self.clustering_algo[m],self.clustering_label[m],2)
                        if m!=1:
                            self.visualise_clusters(dsp_processed_two_components,clustered_tc_data,"Clustered data: "+self.scalar_label[i]+self.two_components_label[j])
                        clustered_mc_data=self.my_model_clusters(dsp_processed_data,self.y,self.clustering_algo[m],self.clustering_label[m],2)
                        x_train, x_test, y_train, y_test = train_test_split(clustered_tc_data, self.y, random_state=0)
                        x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(clustered_mc_data, self.y, random_state=0)
                        for k in range(len(self.supervised_algo)):
                            #Note score has been calculated two times since PCA or NMF produces 2 components & multicomponents in each iteration
                            score=self.my_Classifiers(self.supervised_algo[k],x_train,y_train,x_test,y_test)
                            print("Prediction Accuracy: ",self.scalar_label[i],self.two_components_label[j]+ "Two Components ",self.clustering_label[m],self.supervised_label[k],score)
                            self.r_classifier.append(self.supervised_label[k])
                            self.r_scalar.append(self.scalar_label[i])
                            self.r_cluster.append(self.clustering_label[m])
                            self.r_preprocess.append(self.two_components_label[j])
                            score_m=self.my_Classifiers(self.supervised_algo[k],x_m_train,y_m_train,x_m_test,y_m_test)
                            print("Prediction Accuracy: ",self.scalar_label[i],self.two_components_label[j]+ "Multi Components",self.clustering_label[m],self.supervised_label[k],score_m)
                            self.r_classifier.append(self.supervised_label[k])
                            self.r_scalar.append(self.scalar_label[i])
                            self.r_cluster.append(self.clustering_label[m])
                            self.r_preprocess.append(self.two_components_label[j])

                else:
                    print("NMF can not be applied to non negative matrix of the corresponding scaled data",self.scalar_label[i])
                    continue
    #This method consolidates all these methods to build several models without user intervention
    def full_throttle(self):
        self.supervised_only()
        self.scaled_and_supervised()
        self.scaled_preprocessed_supervised()
        self.clustering_as_classification()
        self.clustering_and_supervised()
        self.scaled_pre_clus_supervised()
        self.call_my_nn()
        self.call_logistic_regression_byhand()
        self.export_report()
    #A neural network of one hidden layer with 16 neurons has been implemented from scratch, more on this at the NN class.
    def call_my_nn(self):
        scaled_data=self.my_Scaler(self.x,self.scalar[0])
        x_train, x_test, y_train, y_test = train_test_split(scaled_data, self.y, random_state=0)

        x=np.asarray(x_train)
        y=np.asarray(y_train)
        x=x.transpose()
        y=y.reshape(1,-1)
        print(x.shape)
        nn=MyNN_From_Scratch(x,y,x.shape[0])
        nn.my_Gradient_Descent(x, y, iter = 1200)
        nn_train=nn.predict(x,y)
        nn_test=nn.predict(np.asarray(x_test).transpose(),np.asarray(y_test).reshape(1,-1))
    #Logistic Regression model built from scratch to compare among other scikit learn based models in terms of performance.
    def call_logistic_regression_byhand(self):
        lg=log_regression_scratch_cv()
        lg.fit(self.x_train,self.y_train)
        score=lg.predict(self.x_test)
        print('Average Score: Normal Scalar cross validated Logistic Regression ',score)
    #This method creates scatter plot of the clusters in two dimentional space, this method is only applied when PCA & NMF returns
    # two dimensional vectors to K-Means & K-Means By Hand classes.
    def visualise_clusters(self,x,pred,title):
        self.count=self.count+1
        color=cm.get_cmap('viridis', 12)
        group1=np.array([])
        group2=np.array([])
        m=0
        print(pred)
        for i in range(len(pred)):
            if pred[i,-1]==0:
                m=m+1
                if group1.shape[0]==0:
                    group1=pred[i,:]
                else:
                    group1=np.vstack((group1,pred[i,:]))
            else:
                if group2.shape[0]==0:
                    group2=pred[i,:]
                else:
                    group2=np.vstack((group2,pred[i,:]))
        #print(group1,m)
        #print(group1[:,0])

        plt.figure(figsize=(10,7))
        #plt.scatter(pred[:, 0], pred[:, 1], c=x[:,-1], cmap=mglearn.cm3, s=7)
        #if len(group1) and len(group2) >1:

        mglearn.discrete_scatter(pred[:, 0], pred[:, 1], pred[:,-1])
        plt.title(title)
 

        #plt.plot(centroids[0,0],centroids[0,1],'rx')
        #plt.plot(centroids[1,0],centroids[1,1],'gx')
        plt.show(block=False)
        plt.pause(2)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        #p=random.randint(100,1000)
        plt.savefig(self.path+'//test'+str(self.count)+'.jpg')
        #plt.savefig(self.path+'//test'+str(p)+'.jpg')
        plt.close()

    #This method creates an autogenerated csv file which compares all the performance metrics of every model which has been applied
    #in this project.
    def export_report(self):
        self.count=self.count+1

        df = pd.DataFrame(list(zip(self.r_scalar,self.r_preprocess,self.r_cluster,self.r_classifier,self.precision, self.recall,self.f1,self.sensitivity,self.specificity)), 
               columns =['Scaling','Preprocessing','Clustering','Classifier', 'Precision', 'Recall','F1 Score','Sensitivity', 'Specificity'])
        print(df)
        df.to_csv(self.path+'//test'+str(self.count)+'.csv')















#preprocess_algo.append(NMF(n_components=2))

#Data applied directly on classifiers


#dummy dataset

#m=my_preprocessing(x,PCA(n_components=2))
#x_train, x_test, y_train, y_test = train_test_split(m, y, random_state=0)
#print(x_train)


#print(my_preprocessing(x_train,PCA_From_Scratch()))


    












        

    







