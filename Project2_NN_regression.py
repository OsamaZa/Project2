# import of the packages needed
import random
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

# To get the same random numbers for every calculation
np.random.seed(95)

# Make data
x = np.arange(0, 1, 0.015)
y = np.arange(0, 1, 0.015)
x,y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    eps = np.random.randn(len(x),1)
    return term1 + term2 + term3 + term4 + 0*eps


z = FrankeFunction(x, y)

# Reshaping to form a X matrix with corresponding x,y values
x = np.reshape(x,(4489,))
y = np.reshape(y,(4489,))
z = np.reshape(z,(4489,))
X=np.zeros((4489,2))
X[:,0]=x
X[:,1]=y


def MSE(y,y_tilde):
    return(np.mean((y - y_tilde)**2))

# Splitting of data into test and training set
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.18734)

# Activation function
def sigmoid(z):
    #The sigmoid function
    return  (1/(1+np.exp(-z)))

# The neural network is based on a code found in lecture notes. This network has 2 hidden layer and 50 nodes in each hidden layer.
class NN:
    # Initializing of the network and the parameters for the regression
    def __init__(self,X_data,Y_data,learning_rate,lmbd,n_hidden_neurons=500,
            epochs=500,batch_size=100):
        self.cost=[]
        self.Epochs=range(epochs)
        self.X_data=np.ravel(X_data)
                
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = 1

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.learning_rate = learning_rate
        self.lmbd = lmbd
        
        self.create_biases_and_weights()

     # Creating the weights and biases
    def create_biases_and_weights(self):
        self.hidden_weights1 = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias1 = np.zeros(self.n_hidden_neurons) + 0.01

        self.hidden_weights2 = np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons)
        self.hidden_bias2 = np.zeros(self.n_hidden_neurons) + 0.01
        
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    # Feed forward for training with two hidden layers and an output layer    
    def feed_forward(self):
        self.z_hidden1 = self.X_data @ self.hidden_weights1 + self.hidden_bias1
        self.a_hidden1 = sigmoid(self.z_hidden1)
    
        self.z_hidden2 = self.a_hidden1 @ self.hidden_weights2 + self.hidden_bias2
        self.a_hidden2 = sigmoid(self.z_hidden2)
        
        self.z_out = self.a_hidden2 @ self.output_weights + self.output_bias

    # Backpropagation where the gradients are made and the weights and biases are updated
    def backpropagation(self):
       
        error_output = (self.z_out - self.Y_data[:,np.newaxis])
        
        self.error = (self.z_out - self.Y_data[:,np.newaxis])
     
        error_hidden2 = (error_output @ self.output_weights.T) * self.a_hidden2 * (1 - self.a_hidden2)
        
        error_hidden1 = (error_hidden2 @ self.hidden_weights2.T) * self.a_hidden1 * (1 - self.a_hidden1)

        self.output_weights_gradient = self.a_hidden2.T @ error_output
        self.output_bias_gradient = np.sum(error_output)

        self.hidden_weights2_gradient = self.a_hidden1.T @ error_hidden2
        self.hidden_bias2_gradient = np.sum(error_hidden2)

        self.hidden_weights1_gradient = self.X_data.T @ error_hidden1
        self.hidden_bias1_gradient = np.sum(error_hidden1)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights1_gradient += self.lmbd * self.hidden_weights1
            self.hidden_weights2_gradient += self.lmbd * self.hidden_weights2

        self.output_weights -= self.learning_rate * self.output_weights_gradient
        self.output_bias -= self.learning_rate * self.output_bias_gradient
        
        self.hidden_weights2 -= self.learning_rate * self.hidden_weights2_gradient
        self.hidden_bias2 -= self.learning_rate * self.hidden_bias2_gradient
        
        self.hidden_weights1 -= self.learning_rate * self.hidden_weights1_gradient
        self.hidden_bias1 -= self.learning_rate * self.hidden_bias1_gradient

    # Feed forward for the predictions
    def feed_forward_out(self,X_data):
        z_hidden1 = X_data @ self.hidden_weights1 + self.hidden_bias1
        a_hidden1 = sigmoid(z_hidden1)

        z_hidden2 = a_hidden1 @ self.hidden_weights2 + self.hidden_bias2
        a_hidden2 = sigmoid(z_hidden2)
        
        z_out = a_hidden2 @ self.output_weights + self.output_bias

        return z_out
    
    # Prediction function
    def predict(self, X):
        z_output = self.feed_forward_out(X)
        return z_output

    # Training function
    def train(self):
        data_indices = np.arange(self.n_inputs)
        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices,size=self.batch_size,replace=False)
                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                
                self.feed_forward()
                self.backpropagation()
            # Saving the error/cost value for each epoch
            self.cost.append(np.mean(self.z_out - self.Y_data[:,np.newaxis]))
        plt.figure()
        plt.plot(self.Epochs,self.cost)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.grid()
        

# Gridsearch for finding the optimal learningrate and penalty parameter
# For linesearch set lmbd = 0 in for loop
learningrates = np.logspace(-8, -2, 4)
lmbds = np.logspace(-8, -2, 4)
test_MSE = np.zeros((len(learningrates), len(lmbds)))
test_R2 = np.zeros((len(learningrates), len(lmbds)))

for i,learn in enumerate(learningrates):
    for j,lmbd in enumerate(lmbds):
        nn=NN(X_train,z_train,learn,lmbd)
        nn.train()
        test_predict=nn.predict(X_test)
        print("Learning rate  = ", learn)
        print("Lambda = ", lmbd)
        print("MSE on test set: ", MSE(z_test, test_predict))
        print("R2 on test set: ", r2_score(z_test, test_predict))
        test_MSE[i][j] = MSE(z_test, test_predict)
        test_R2[i][j] = r2_score(z_test, test_predict)

        # Plotting the surface from the trained network
        z_predict = nn.predict(X)
        z_predict = z_predict.reshape(67,67)
        X_x = X[:,0].reshape(67,67)
        X_y = X[:,1].reshape(67,67)
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X_x, X_y, z_predict, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

# Making a heatmap of the MSE's
fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(test_MSE, annot=True, ax=ax)
ax.set_title("Test MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

# Making a heatmap of the R2_scores
fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(test_R2, annot=True, ax=ax)
ax.set_title("Test R2")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

# Plotting the orignal surface
X_x = X[:,0].reshape(67,67)
X_y = X[:,1].reshape(67,67)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_x, X_y, z_predict, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


mlp=MLPRegressor(hidden_layer_sizes=(50,2), activation='relu', solver='sgd')
mlp.fit(X_train,z_train)

test_predict=mlp.predict(X_test)

print("MSE on test set: ", MSE(z_test, test_predict))
print("R2 on test set: ", r2_score(z_test, test_predict))



plt.show()
