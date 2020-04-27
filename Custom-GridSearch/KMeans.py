import numpy as np
import math
import scipy.cluster as sc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

class kmeansByHand:
    def __init__(self,k,path):
        self.k=k
        self.labels_=[]
        self.path=path
    # calculating euclidean distance
    def dist(self,p1, p2):
        sumTotal = 0

        for c in range(len(p1)):
            sumTotal = sumTotal + pow((p1[c] - p2[c]),2)

        return math.sqrt(sumTotal)

    def minDistPos(self,point, matrix):
        minPos = -1
        minValue = float("inf")

        for rowPos in range(len(matrix)):
            d = self.dist(point,matrix[rowPos,:])

            if (d < minValue):
                minValue = d
                minPos = rowPos

        return minPos

            
    #sum of euclidean distance acroos all dimensions
    def sumDist(self,m1, m2):
        sumTotal = 0
        
        for pos in range(len(m1)):
            sumTotal = sumTotal + self.dist(m1[pos,:],m2[pos,:])

        return sumTotal
    #standardising the data 
    def standard(self,data):
        standardData = data.copy()
        
        rows = data.shape[0]
        cols = data.shape[1]
        #print(cols)
        #print(data)

        for j in range(cols):
            sigma = np.std(data[:,j])
            mu = np.mean(data[:,j])

            for i in range(rows):
                standardData[i,j] = (data[i,j] - mu)/sigma

        return standardData
    def fit_predict(self,x):
        reducedData = np.asarray(x).astype(np.float)
        #y=data[:,8:]
        standardisedData = self.standard(reducedData)

        columns = x.shape[1]

        # Number of clusters
        k = self.k

        # Initial centroids
        C = np.random.random((k,columns))

        # scale the random numbers
        for f in range(columns):
            maxValue = standardisedData[:,f].max()
            minValue = standardisedData[:,f].min()

            for c in range(k):
                C[c,f] = minValue + C[c,f] * (maxValue - minValue)
            
        C_old = np.zeros(C.shape)

        clusters = np.zeros(len(standardisedData))

        distCentroids = float("inf")

        threshold = 0.1

        while distCentroids > threshold:
            for i in range(len(standardisedData)):
                clusters[i] = self.minDistPos(standardisedData[i], C) #finding points that has a minimum distance with the centroid

            C_old = C.copy()

            for i in range(k):
                points = np.array([])

                for j in range(len(standardisedData)):
                    if (clusters[j] == i):   #points with mimimum distance are considered within the cluster
                        if (len(points) == 0):
                            points = standardisedData[j,:].copy()
                        else:
                            points = np.vstack((points,standardisedData[j,:]))

                C[i] = np.mean(points, axis=0) # finding mean of the points that were assigned to the same cluster
                
            distCentroids = self.sumDist(C, C_old) 

            
        centroids = C  #centroids were shifted to the mean of the clusters
        #print(centroids)
        group1 = np.array([])

        group2 = np.array([])
        pred=[]
        for d in standardisedData:
            if (self.dist(d, centroids[0,:]) < self.dist(d, centroids[1,:])):  #comparing the distance between a point & the clusters 
                #& assigning it to the corresponding cluster that has minimum distance & the process is repeated.
                pred.append(1)
                if (len(group1) == 0):
                    group1 = d
                    
                else:
                    group1 = np.vstack((group1,d))
                    
            else:
                pred.append(0)
                if (len(group2) == 0):
                    group2 = d
                else:
                    group2 = np.vstack((group2,d))
        self.labels_=np.asarray(pred)
        print("Group 2",group2.shape)

        return pred




#plt.savefig("kmeansByHandClassified.pdf")

#plt.close()
