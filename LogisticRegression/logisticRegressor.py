from LogisticRegression.NormalEquation import normalEquation
from LogisticRegression.GradientDescent import gradientDescent
import numpy as np
class LogisticRegressor():
    def __int__(self):
        self.X = None
        self.y = None
        self.theta = None
    def fit(self,X,y):
        self.X = X.values
        self.y = y.values
        sample,features = X.shape
        self.used = None
        #UnBiased Provides a Better Model For Logistics Regression when Normal Equations are Used (53% - 84%)
        if(sample<100):
            self.used = 1
            self.theta = normalEquation.linearNormalEquation(self.X,self.y.reshape(len(self.y),1))
            print(f"Theta using Normal Equations:{self.theta}")
        else:
            self.used = 2
            ones = np.ones((sample,1),dtype='i4')
            self.X = np.append(ones,self.X,axis=1)
            self.theta,self.cost_history,self.theta_history = gradientDescent.gradient_Descent(self.X,self.y,np.zeros(features+1))
            print(f"Theta using Gradient Equations:{self.theta}")
    def predict(self,X):
        samples,cols = X.shape
        sample_X = X
        #Use Bias Model When Gradient Descent is Used
        if(self.used == 2):
            ones = np.ones((samples,1),dtype='i4')
            sample_X = np.append(ones,X,axis=1)
        predicts = np.dot(sample_X,self.theta)
        predicts = self.sigmoid(predicts)
        return predicts
    def sigmoid(self,predict):
        predicts = list()
        for pr in predict:
            if((1/(1 + (np.e**(-pr))))>=0.5):
                predicts.append(1)
            else:
                predicts.append(0)
        return np.array(predicts)