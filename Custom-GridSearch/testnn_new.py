import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class MyNN_From_Scratch:
    def __init__(self, x, y, d):
        #initializing tthe variables
        self.X=x
        self.Y=y
        self.N_output=np.zeros((1,self.Y.shape[1]))
        self.L=2
        self.dimension = [d, 16, 1]
        self.parameters = {}
        self.temp = {}
        self.loss = []
        self.learning_rate=0.01
        self.t_samples = self.Y.shape[1]
        self.threshold=0.5
    def weight_Initialize(self): 
        #weights of the layers are initialised here   
        np.random.seed(1)
        self.parameters['W1'] = np.random.randn(self.dimension[1], self.dimension[0]) / np.sqrt(self.dimension[0]) 
        self.parameters['b1'] = np.zeros((self.dimension[1], 1))        
        self.parameters['W2'] = np.random.randn(self.dimension[2], self.dimension[1]) / np.sqrt(self.dimension[1]) 
        self.parameters['b2'] = np.zeros((self.dimension[2], 1))                
        return
    def Sigmoid(self,Z):
        #Our Activation function at the output layer
        return 1/(1+np.exp(-Z))
    def Relu(self,Z):
        #Our activation function at the hidden layer
        return np.maximum(0,Z)
    def forward_Prop(self):    
        Z1 = self.parameters['W1'].dot(self.X) + self.parameters['b1'] 
        #Activation function applied on weights & biases
        A1 = self.Relu(Z1)
        self.temp['Z1'],self.temp['A1']=Z1,A1
        #Output layer which is taking A1 as the input & multiplying weights & biases
        Z2 = self.parameters['W2'].dot(A1) + self.parameters['b2']  
        #Sigmoid activation function has been used at the end of the output layer
        A2 = self.Sigmoid(Z2)
        self.temp['Z2'],self.temp['A2']=Z2,A2

        self.N_output=A2
        #print(A2)
        #calculating loss for the forward pass
        loss=self.calculate_loss(A2)
        return self.N_output, loss
    def calculate_loss(self,N_output):
        #c=np.log(1-N_output).T
        #print(c)
        #print("Sample size",self.sam)
        
        loss = (1./self.t_samples) * (-np.dot(self.Y,np.log(N_output).T) - np.dot(1-self.Y, np.log(1-N_output).T))  
        #print("loss",loss)  
        return loss
    def dRelu(self,x):
        #Gradient function of Relu
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def dSigmoid(self,Z):
        #Gradient function of Sigmoid
        s = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ
    def backward_Prop(self):
        dLoss_N_output = - (np.divide(self.Y, self.N_output ) - np.divide(1 - self.Y, 1 - self.N_output))    
        #calculating the loss accross the network by using chainrule of partial derivative accross each layer of the neural network
        #Gradient of sigmoid has been used calculated at the output layer to form this equation
        dLoss_Z2 = dLoss_N_output * self.dSigmoid(self.temp['Z2'])    
        dLoss_A1 = np.dot(self.parameters["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.temp['A1'].shape[1] * np.dot(dLoss_Z2,self.temp['A1'].T)
        dLoss_b2 = 1./self.temp['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
        #Gradient of  Relu has been applied to this equation to find the equation for cumulative loss         
        dLoss_Z1 = dLoss_A1 * self.dRelu(self.temp['Z1'])        
        dLoss_A0 = np.dot(self.parameters["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        #Now each parameter has been modified by a very small learning rate to measure the change in it's gradient functions to overall loss
        self.parameters["W1"] = self.parameters["W1"] - self.learning_rate * dLoss_W1
        self.parameters["b1"] = self.parameters["b1"] - self.learning_rate * dLoss_b1
        self.parameters["W2"] = self.parameters["W2"] - self.learning_rate * dLoss_W2
        self.parameters["b2"] = self.parameters["b2"] - self.learning_rate * dLoss_b2

        return
    
    #Measures simple accuracy by checking labels
    def predict(self,x, y):  
        self.X=x
        self.Y=y
        comp = np.zeros((1,x.shape[1]))
        pred, loss= self.forward_Prop()    
    
        for i in range(0, pred.shape[1]):
            if pred[0,i] > self.threshold: comp[0,i] = 1
            else: comp[0,i] = 0
    
        print("Acc: " + str(np.sum((comp == y)/x.shape[1])))
        
        return comp

    def my_Gradient_Descent(self,X, Y, iter = 3000):

        np.random.seed(1)                         
        
        self.weight_Initialize()
        #For each epoch the backpropagation provides the loss function of the network with a small change in learning rate
        #This process is repeated untill the error doesn't have any effect on varying the learning rate or when the convergence is reached.
        for i in range(0, iter):
            N_output, loss=self.forward_Prop()
            #print("Loss after Fpass",loss)
            self.backward_Prop()
            
            if i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, loss))
                self.loss.append(loss)
    
        return



