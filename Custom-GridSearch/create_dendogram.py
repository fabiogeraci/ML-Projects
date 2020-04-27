import numpy as np
import math
import scipy.cluster as sc
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt

class my_dendogram:

#Calculates euclidean distance between two points in multi dimensional space
    def distance(self,data):

        rows = data.shape[0]
        cols = data.shape[1]
        
        distanceMatrix = np.zeros((rows,rows))

        for i in range(rows):
            for j in range(rows):

                sumTotal = 0

                for c in range(cols):

                    sumTotal = sumTotal + pow((data[i,c] - data[j,c]),2)

                distanceMatrix[i,j] = math.sqrt(sumTotal)

        return distanceMatrix
#This method creates a tree based dendrogram which can help decide how many clusters can be useful for that particular dataset
# & the number k can be provided as a parameter to my_tools class which in turn passes it to K-Means algorithm.
    def initiate(self,data,title):


        distanceData = self.distance(data)

        condensedDistance = sd.squareform(distanceData)

        Z = sc.hierarchy.linkage(condensedDistance)
        plt.figure(figsize=(6,4))
        plt.axhline(y=.5, ls='--')
        #ax.plot(bounds, [5, 5], '--', c='k')
        
        plt.xlabel("Sample index")
        plt.ylabel("Cluster distance")
        plt.title(title)
        sc.hierarchy.dendrogram(Z, truncate_mode="lastp",p=10)
        

        plt.show()

        