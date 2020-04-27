import numpy as np
import random
import math


class handle_imbalance:
    def __init__(self):
        self.temp=[]
        self.temp1=np.array([])
        self.temp2=np.array([])
    #Mean based approach to balance between under sampling & oversampling
    def simple_oversample_undersample(self,data,y):
        new_data=data.copy()
        n_data=data.copy()
        print('oversample running')
        #new_data_x=data.copy()
        label,count=np.unique(y,return_counts=True)
        major_class_tuple=[label[np.argmax(count)],np.max(count)] #class that has highest frequency
        minor_class_tuple=[label[np.argmin(count)],np.min(count)] # the class with less frequency
        sum=major_class_tuple[1]+minor_class_tuple[1]
        opt_sample=int(sum/2)  #mean sample size.... thats the target for both
        
        #Let's oversample now
        k=opt_sample-minor_class_tuple[1]
        print('k',k)
        self.temp=k,minor_class_tuple[0]
        labelpos=[]
        #Find all positions of the minor class
        for rowPos in range(len(y)):
            if y[rowPos]==minor_class_tuple[0]:
                labelpos.append(rowPos)

        #Now duplicate k number of indices with replacement
        for n in range(k):
            index=labelpos[random.randint(0,len(labelpos)-1)] #each row is chosen randomly from the dataset
            item=data[index]
            new_data=np.vstack((new_data,item)) 
        
        new_data=new_data[new_data[:,-1]==minor_class_tuple[0]] #filtering only the minor class labels
        n_data=n_data[n_data[:,-1]==minor_class_tuple[0]]  #it is same as new_data which can be used when working with SMOTE
        print('new data',new_data.shape)
       # print('n_data ',n_data.shape)
        #new_data=np.vstack((new_data,n_data)
        self.temp1=n_data
        #print('oversampled new data',new_data.shape)



        #Let's undersample the major class now
        newlabelpos=[]
        #finding indices with major class
        for i in range(len(y)):
            if y[i]==major_class_tuple[0]:
                newlabelpos.append(i)

        k_new=major_class_tuple[1]-opt_sample
        #let's eliminate k samples
        elm=random.sample(newlabelpos,k_new)
        #print(np.asarray(elm).shape)
        #updating the new data by eliminating the selected rows from the list elm
        new_data_x=np.array([])
        for rowpos in range(len(data)):
            if(rowpos not in elm):
                if len(new_data_x)==0:
                    new_data_x=data[rowpos,:].copy()
                    #print(new_data)
                else:
                    #print(data[rowpos,:])
                    new_data_x=np.vstack((new_data_x,data[rowpos,:]))
        

        new_data_x=new_data_x[new_data_x[:,-1]==major_class_tuple[0]] #filtering the data with major class only
        self.temp2=new_data_x
        print('new data x', new_data_x.shape)
        final_data=np.vstack((new_data,new_data_x)) #finally merging both the undersampled & oversampled data together
        print('final data ',final_data.shape)
        return final_data

    #Euclidian distance between the data points
    def calculate_dist(self,p1,p2) :
        sum_of_distances = 0
        for c in range(len(p1)):
            sum_of_distances=sum_of_distances + pow((p1[c]-p2[c]),2)
        return math.sqrt(sum_of_distances)
    #This method is same as the previous one except it's using SMOTE to generate synthetic data for oversampling rather duplicating
    #previous data points
    def modified_smote(self,data,y,k_neighbors):

        synthetic_data=data[1,:].copy()
        #dummycall=self.simple_oversample_undersample(data,y)
        sample_size=self.temp[0]
        minor_class=self.temp[1]
        labelpos=[]
        #finding all indices for the minor class
        for i in range(len(y)):
            if y[i]==minor_class:
                labelpos.append(i)
        print('sample size',sample_size)

        for iteration in range(sample_size):
            #choosing an item with minor class label
            item_x=labelpos[random.randint(0,len(labelpos)-1)]
            #now using k nearest neighbors to calculate the points with minimum distance from this point
            k_dist=[]
            index_dist=[]
            for item in labelpos:
                if item==item_x:
                    continue
                k_dist.append(self.calculate_dist(data[item],data[item_x]))
                index_dist.append(item)
            knn=[]
            for k in range(k_neighbors):
                closest=np.argmin(k_dist)
                knn.append(index_dist[closest])
                k_dist[closest]=float("inf")
            #randomly choosing one from the k neighbors
            item_y=knn[random.randint(0,len(knn)-1)]
            #calculating the vector between the points for every feature
            vec=[]
            for i in range(data.shape[1]):
                vec.append(data[item_y,i] - data[item_x,i])
            r=random.random()
            syn_data= [(data[item_x,i]+r*vec[i]) for i in range(data.shape[1])] #finding a random points in between the initial point & it's randomly selected nearest neighbor
            synthetic_data=np.vstack((synthetic_data,syn_data))
            #print(syn_data)
            #print('running')
        temp_data=np.vstack((synthetic_data,self.temp1)) #copied major class data after undersampling
        final_data=np.vstack((temp_data,self.temp2)) #finally synthetic data is merged with undersampled data
        print(final_data.shape)
        return final_data







                







