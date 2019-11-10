# import of the packages needed
import random
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

# To get the same random numbers for every calculation
np.random.seed(95)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Making a heatmap of the correlation between the features
plt.figure()
corr = df.corr()
ax = sns.heatmap(corr)
plt.title("Heatmap of the correlation matrix")
# Making a horisontal barplot to illustrate the correlations to default payment
plt.figure()
cor=corr["defaultPaymentNextMonth"]
cor.drop(cor.index[-1]).plot.barh()
plt.title("Correlation values for 'default payment'")
# Printing out the values of the correlation between the features and 'default payment'
print("_____Correlation to default payment_____")
print(cor.drop(cor.index[-1]))
df = df.drop(df[(df.AGE == 0) &
                (df.MARRIAGE == 0) &
                (df.EDUCATION == 0) &
                (df.SEX == 0) &
                (df.LIMIT_BAL == 0)].index)

# From the horisontal barplot the correlation for BILL_AMT5 and BILL_AMT6 is small and therefore dropped.
df=df.drop(['BILL_AMT5', 'BILL_AMT6'], axis=1)

# Including the education 5 and 6 in group 4(other)
df['EDUCATION']=np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])


# Features and targets 
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

# Input scaling
sc = StandardScaler()
X[:,[0,4,11,12,13,14,15,16,17,18,19,20]] = sc.fit_transform(X[:,[0,4,11,12,13,14,15,16,17,18,19,20]])

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer([("",onehotencoder,[1,2,3,5,6,7,8,9,10]),],remainder="passthrough").fit_transform(X)

y=np.ravel(y)

# Splitting of data into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Making a balanced dataset with indexes
y_tr_index=[]
for i in range(len(y_train)):
    if y_train[i]==1:
        y_tr_index.append(i)
        
y_tr_index0=[]
i=0
while len(y_tr_index0)<len(y_tr_index):
    if y_train[i]==0:
        y_tr_index0.append(i)
        i+=1
    else:
        i+=1
# Indexes for the balanced dataset       
nonbiased_data_indexes=(y_tr_index0+y_tr_index)
np.random.shuffle(nonbiased_data_indexes)

# Activation function
def sigmoid(z):
    return  (1/(1+np.exp(-z)))

# The neural network is based on a code found in lecture notes. This network has 2 hidden layer and 50 nodes in each hidden layer.
class NN:
    # Initializing of the network and the parameters for the regression
    def __init__(self,X_data,Y_data,learning_rate,lmbd,n_hidden_neurons=50,
            epochs=50,batch_size=100):
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
    
        self.probabilities = (sigmoid(self.z_out))

    # Backpropagation where the gradients are made and the weights and biases are updated
    def backpropagation(self):
       
        error_output = (self.probabilities - self.Y_data[:,np.newaxis]) * self.probabilities * (1-self.probabilities)
        
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
        
        probabilities = (sigmoid(z_out))
        
        probabilities[probabilities >= 0.5] = 1
        probabilities[probabilities < 0.5] = 0
        
        return probabilities

    # Prediction function
    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

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
           
learningrates = np.logspace(-8, -2, 10)
lmbd=0
for learn in learningrates:
    nn=NN(X_train[nonbiased_data_indexes,:],y_train[nonbiased_data_indexes],learn,lmbd)
    nn.train()
    test_predict=nn.predict(X_test)
    print("Learning rate  = ", learn)
    # SciKitLearn neuralnetwork prediction
    MLP = MLPClassifier(solver='sgd',hidden_layer_sizes=(50,50))
    MLP.fit(X_train[nonbiased_data_indexes,:],
            y_train[nonbiased_data_indexes])
    sci_test_predict=MLP.predict(X_test)

    # Accuracy score for scikit prediction
    print("Accuracy score on test set with SciKit: ", accuracy_score(y_test, sci_test_predict))
    print("Accuracy score on test set: ", accuracy_score(y_test, test_predict))

    # Making the confusion matrix for both predictions
    SciKit_CM = confusion_matrix(y_test, sci_test_predict)
    print("___SciKit___")
    print("Right guessed 0:", SciKit_CM[0,0])
    print("Right guessed 1:", SciKit_CM[1,1])
    print("Wrong guessed 0:", SciKit_CM[0,1])
    print("Wrong guessed 1:", SciKit_CM[1,0])

    Own_CM = confusion_matrix(y_test, test_predict)
    print("___Own code___")
    print("Right guesses 0:", Own_CM[0,0])
    print("Right guesses 1:", Own_CM[1,1])
    print("Wrong guesses 0:", Own_CM[0,1])
    print("Wrong guesses 1:", Own_CM[1,0])
    
plt.show()
       
