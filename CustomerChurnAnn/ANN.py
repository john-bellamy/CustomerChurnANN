# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 23:48:51 2018

@author: johnb
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#For the Neural Network
from keras.models import Sequential
from keras.layers import Dense


#Testing
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Importing the dataset
dataset = pd.read_csv('DATA\\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Drop one country column
X = X[:,1:]
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Drop one country column
X = X[:,1:]


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling Always

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Create your classifier here

# We will use method one and define input layers and output layer

classifier = Sequential()

#Step 1 initialize weights to zero (kernel_initializer)
#2 Create input layer and output layer
#3 Choose activation function. relu = rectifier function
#4 Choose number of nodes in input layer

#Tip : Number of nodes in hidden layer can be average of nodes in input layer
# and output layer. So in this case, we take 11 independent variables plus
# one independent variable divded by two and we get 6.

#Adds input layer and first hidden layer

classifier.add(Dense(units=6, kernel_initializer='uniform', 
                     activation='relu', input_dim=11))

# Not necessary, but add second hidden layer like this:

classifier.add(Dense(units=6, kernel_initializer='uniform', 
                     activation='relu'))

# Add output layer
#For dependent variabel with three categories, you would change output_dim
#to three and activation to softmax

classifier.add(Dense(units=1, kernel_initializer='uniform', 
                     activation='sigmoid'))

#Compile network: this appleid stochastic gradient descent
#Your loss function should match the activation function for the output
#More than two category is categorical_crossentropy

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

#Fitting ANN to training set

 
# Predicting the Test set results
y_pred = classifier.predict(X_test)

#We need to convert probabilities into true or false values, with a threshhold

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

#Accuracy
# total correct / number in training set
accuracy = (1548 + 136) / (2000)

#Predict a new item
#two brackets create a 2d array
# Remember we need to encode categories and scale newdata
new_prediction = classifier.predict(sc.fit_transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > .5)

new_prediction
# Model Selection:

def classifier_builder(optimizer):
    
    classifier = Sequential()

    classifier.add(Dense(units=6, kernel_initializer='uniform', 
                         activation='relu', input_dim=11))
    
    
    classifier.add(Dense(units=6, kernel_initializer='uniform', 
                         activation='relu'))
    
    
    classifier.add(Dense(units=1, kernel_initializer='uniform', 
                         activation='sigmoid'))
    
    
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics= ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = classifier_builder)

# create dictionary with hyper-parameters (hyper-parameterselection)
#The dictionary keys needto match parameters in function
#Try powers of two
parameters = {'batch_size': [128,256,512,1048], 'epochs' : [10,20,40,50,100,300],
              'optimizer': ['adam','rmsprop']}

# Grid search will use k-fold
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,
                           scoring = 'accuracy', cv = 10)

# Now we need to fit grid_search to data
grid_search = grid_search.fit(X_train, y_train)

#Get the best parameters
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_


print("Best accuracy is: %.2f" % best_accuracy)

print("Best batch-size: %d" % best_parameters["batch_size"])


print("Best number of epochs: %d" % best_parameters["epochs"])


print("Best optimizer: %s" % best_parameters["optimizer"])


classifier = Sequential()

#Add input layer
classifier.add(Dense(units=6, kernel_initializer='uniform', 
                     activation='relu', input_dim=11))


classifier.add(Dense(units=6, kernel_initializer='uniform', 
                     activation='relu'))

# Add output layer


classifier.add(Dense(units=1, kernel_initializer='uniform', 
                     activation='sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics= ['accuracy'])


#Fitting ANN to full set with best parameters

X_full = np.vstack([X_train, X_test])
classifier.fit(X_full,y, batch_size=256, epochs=300)

new_prediction = classifier.predict(sc.fit_transform(np.array([[0,0,600,1,40,3,65000,2,1,1,45000]])))
new_prediction = (new_prediction > .5)

new_prediction
    
