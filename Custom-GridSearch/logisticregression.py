import pandas as pd
import numpy as np
from math import exp
from random import randrange
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer

#Please check the dataset in the folder or use a different dataset
#If you are using the pulser dataset please be patient
class LogisticRegression_scratch:

    #Accuracy function to check the accuracy of the model
    def accuracy_score(self, true_value, prediction):
        match = 0
        for i in range(len(true_value)):
            if true_value[i] == prediction[i]:  #finding the matches between the predicted values & true values of labels
                match += 1
        return match / float(len(true_value)) * 100.0  #calculation of simple accuracy with this formula

    #sigmoid function that returns the sigma(x) for each row of training data
    def sigma(self, row, Beta):  
        y = Beta[0]  #bias term or Beta(0) of the equation
        for i in range(len(row)-1):
            y += Beta[i + 1] * row[i] # this equation is equivalent to s=b1*x1 + b2*x2 +..... bi*xi
        return 1.0 / (1.0 + exp(-y)) #returns the sigma of x

    #Stochastic gradient descent function to minimize the loss in each iteraction for every randomly taken batch training data
    def stochastic_gradient_descent(self, training_data, l_rate, n_epoch):
        Beta = [0.0 for i in range(len(training_data[0]))] #Initialisation of all coefficients of x which is b0,b1,...bi
        #print(Beta)
        for epoch in range(n_epoch):  #number of iterations for which training needs to be done for minimising loss
            for row in training_data:
                sigma_x = self.sigma(row, Beta)  #returns sigma of x
                #print(sigma_x)
                error = row[-1] - sigma_x #calculating loss after each row of trained data
                #print('error ',error)
                Beta[0] = Beta[0] + l_rate * error * sigma_x * (1.0 - sigma_x) #updating the coefficient Beta(0)
                #print(Beta)
                for i in range(len(row)-1): # updating every other coefficient other than bias or b0; that is b1,b2...bi
                    Beta[i + 1] = Beta[i + 1] + l_rate * \
                        error * sigma_x * (1.0 - sigma_x) * row[i]
        return Beta

    #logistic regression function to determine the predictions 
    def logistic_regression(self, training_data, test, l_rate, n_epoch):
        predictions = list()
        Beta = self.stochastic_gradient_descent(training_data, l_rate, n_epoch) #returns the optimal value of coefficients after training 
        for row in test:
            sigma_x = self.sigma(row, Beta) #sigma of x is determined for each row of test data & coefficients returned from training
            sigma_x = round(sigma_x)
            predictions.append(sigma_x)
        return(predictions)

class log_regression_scratch_cv:
#data input using panads dataframe
    def fit(self,x,y):
        self.x=x
        self.y=y
    def predict(self,x):
        #input_dataset = pd.read_csv("HTRU_2.csv")
        #print(input_dataset.shape)
        y=self.y
        input_dataset=self.x
        n_batches = 5 #number of batches here is hardcoded but can be modified anytime
        nm = Normalizer()
        data = nm.fit_transform(input_dataset) #normalizing the data
        data=np.column_stack((data,y))
        batch_data = list()
        batch_data_c = list(data)
        batch_size = int(len(data) / n_batches) #number of batches of data to be sampled for training
        for i in range(n_batches):  #for each batches choosing the data randomly from the entire dataset
            batch = list()
            while len(batch) < batch_size:
                index = randrange(len(batch_data_c)) #randomly choosing rows from the entire data till it reaches the batch size
                #print(index)
                batch.append(list(batch_data_c.pop(index)))
                #print(batch)
                #print('breakpoint')
            batch_data.append(batch)
        scores = list()
        lr = LogisticRegression_scratch()
        for batch in batch_data:
            training = list(batch_data) #training data is created from the list of batches
            #print(batch)
            training.remove(batch)
            training = sum(training, []) #this creates the training data of n-1 batches so that the other batch is used for testing
            #print(training)
            test = list()
            for element in batch:
                element_copy = list(element)
                test.append(element_copy) #testing data for each batches
                element_copy[-1] = None  #removing label from test set
            prediction = lr.logistic_regression(training, test, .001, 100) # here learning rate is .001 & epoch size is 100
            true_value = [element[-1] for element in batch]  #this is actually y_test or lables of test data
            print(classification_report(true_value,prediction))  #Detailed report of precision recall & f1 score for each class & overall data
            accuracy = lr.accuracy_score(true_value, prediction) #Simple accuracy metric to measure performance
            scores.append(accuracy)
        print(scores)
        avg=sum(scores)/len(scores)
        print('Average score',avg )
        return avg

