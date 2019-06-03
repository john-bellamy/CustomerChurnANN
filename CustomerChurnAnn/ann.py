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

X = pd.read_csv('data/x_out.csv')
X = X.ix[:,1:]

y =  pd.read_csv('data/y_out.csv', names=['y'] )
y = y.ix[:,0:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

def classifier_builder(optimizer):
    
    classifier = Sequential()

    classifier.add(Dense(units=6, kernel_initializer='uniform', 
                         activation='relu', input_dim=7))
    
    
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