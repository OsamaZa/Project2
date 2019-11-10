# import of the packages needed
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns

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

# Printing out the values of the correlation between the features and the targets
print("_____Correlation to targets_____")
print(cor.drop(cor.index[-1]))

# Dropping the rows where the requirement in the paranthesis is fulfilled
df = df.drop(df[(df.AGE == 0) &
                (df.MARRIAGE == 0) &
                (df.EDUCATION == 0) &
                (df.SEX == 0) &
                (df.LIMIT_BAL == 0)].index)

# Dropping the two features with lowest correlation with the default payment
df=df.drop(['BILL_AMT5', 'BILL_AMT6'], axis=1)
            
# Putting values for education to 4 when originally equal to 5 and 6
df['EDUCATION']=np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])

# Features and targets
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

# Input Scaling
sc = StandardScaler()
X[:,[0,4,11,12,13,14,15,16,17,18,19,20]] = sc.fit_transform(X[:,[0,4,11,12,13,14,15,16,17,18,19,20]])

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer([("",onehotencoder,[1,2,3,5,6,7,8,9,10]),],remainder="passthrough").fit_transform(X)

# Making y a contiguous flattened array
y=np.ravel(y)

# Splitting the data into training data and test data
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

# Defining the sigmoid function
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# Defining the cost function
def cost(beta, X, y,lmbd):
    predictions = sigmoid(X @ beta)
    predictions[predictions == 1] = 0.999999 # log(1-p) causes error during optimization
    cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions)+ lmbd*(beta.T @ beta))
    return cost

# Defining the gradient of the cost function
def cost_gradient(beta, X, y,lmbd):
    predictions = sigmoid(X @ beta)
    return (X.T @ (predictions-y) + lmbd*beta)

#THIS PART IS FOR PLOTTING OF THE COST FUNCTION/ERROR AND ALSO FOR LINESEARCH
costs=[]
costgrad=[]
iterations = 500
#learn=1e-05 #FOR THE COST FUNCTION PLOT
learningrates = np.logspace(-8, -2, 10) #FOR THE LINESEARCH
lmbd=0.0001
learn=0.0001
#for learn in learningrates: # FOR THE LINESEARCH
beta = np.random.randn(np.shape(X_train[nonbiased_data_indexes,:])[1])
for itr in range(iterations):
    beta -= learn*cost_gradient(beta,X_train[nonbiased_data_indexes,:],y_train[nonbiased_data_indexes],lmbd)
    costs.append(cost(beta,X_train[nonbiased_data_indexes,:],y_train[nonbiased_data_indexes],lmbd))
    costgrad.append(cost_gradient(beta,X_train[nonbiased_data_indexes,:],y_train[nonbiased_data_indexes],lmbd))
"""
    # FOR THE LINESEARCH
    t = X_test @ beta
    predictions = np.zeros(len(y_test))
    predictions[sigmoid(t) >= 0.5] = 1
    predictions[sigmoid(t) < 0.5] = 0
    print("Learning rate  = ", learn)
    print("Test set accuracy for own code=", accuracy_score(y_test,predictions))
    Own_CM = confusion_matrix(y_test, predictions)
    print("___Own code___")
    print("Right guesses 0:", Own_CM[0,0])
    print("Right guesses 1:", Own_CM[1,1])
    print("Wrong guesses 0:", Own_CM[0,1])
    print("Wrong guesses 1:", Own_CM[1,0])
    # Making a prediction based on the scikit learn logistic regression
    logreg = LogisticRegression(solver="saga",penalty='l1',max_iter=1000)
    logreg.fit(X_train[nonbiased_data_indexes,:], y_train[nonbiased_data_indexes])
    y_pred = logreg.predict(X_test)
    SciKit_CM = confusion_matrix(y_test, y_pred)
    print("Test set accuracy for SciKit", accuracy_score(y_test,y_pred))
    print("___SciKit___")
    print("Right guessed 0:", SciKit_CM[0,0])
    print("Right guessed 1:", SciKit_CM[1,1])
    print("Wrong guessed 0:", SciKit_CM[0,1])
    print("Wrong guessed 1:", SciKit_CM[1,0])
    ########
"""    
# FOR THE COST FUNCTION PLOT   
# Plotting the cost function as a function of iterations
plt.figure()    
plt.plot(range(iterations),costs)
plt.title("Cost function as a function of number of iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.grid()

# Plotting the gradient of the cost function as a function of iterations
plt.figure()
plt.plot(range(iterations),costgrad)
plt.title("Gradient of the cost function as a function of number of iterations")
plt.xlabel("Iterations")
plt.ylabel("Gradient of cost function")
plt.grid()
########


plt.show()
