"""""
Creating a new class called Logistic Regression which inherits from LinearModel.
Class has two methods:
- LogisticRegression.loss(X,y) that computes the empirical risk L(w) using logisitc loss function  
    - weight vector w used should be stored as an instance var of the class
- LogisticRegression.grad(X,y) that computes the gradient of the empirical risk L(W)
"""""
import torch
### from class warmup 
class LinearModel:

    def __init__(self):
        # self.w is the current weight w_k
        #self.w_prev is the old weight vector w_(k-1)
        self.w = None 
        self.w_prev = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s 
        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        S = self.score(X)
        y_hat = 1.0 * (S >= 0)
        return y_hat

class LogisticRegression(LinearModel):
    def loss(self, X, y):
        """"" 
        Computes the rate of misclassification (loss)

        ARGUMENTS:
         X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.
         y, torch.Tensor: the target vector. y.size() == (n,). vector predictions in {0.0, 1.0}

        RETURNS:
         value of L(w), torch.Tensor. L(w).size() == (1), the loss value
         """
        #defining sigmoid value
        sig = torch.sigmoid(self.score(X))

        #calculating loss
        loss = -y*(torch.log(sig)-(1-y)*torch.log(1-sig))
        return torch.mean(loss)
    
    def grad(self, X, y):
        """""
        Computes the gradient of the empirical risk L(w)

        ARGUMENtS:
         X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.
         y, torch.Tensor: the target vector. y.size() == (n,). vector predictions in {0.0, 1.0}

        RETURNS:
         grad, float, the value for gradient- greatest loss
        """""
                                        #convering tensor with shape (n,) to (n,1)
        grad = (torch.sigmoid(self.score(X))-y)[:, None]
        return torch.mean(grad*X, dim = 0)
    
class GradientDescentOptimizer(LogisticRegression):
    #Gradient descent with momentum, aka Spicy Gradient Descent
    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha, beta):
        """""
        Computes a step of the logistic regression, updating using feature matrix X and target vector y
    
        ARGUMENTS:
        X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.
        y, torch.Tensor: the target vector. y.size() == (n,). vector predictions in {0.0, 1.0}
        alpha, float, the learning rate
        beta, float, the momentum

        RETURNS:
        value of L(w), torch.Tensor. L(w).size() == (1), the loss value at that step
        """""
        grad = self.model.grad(X,y)
        loss = self.model.loss(X,y)
        weight = self.model.w

        # for the first update:
        if self.model.w_prev == None:
            self.model.w -= alpha*grad
        
        #> first update:
        else:
            self.model.w = weight - 1*alpha*grad+beta*(weight - self.model.w_prev)
        
        self.model.w_prev = weight

        return loss
    


        
        