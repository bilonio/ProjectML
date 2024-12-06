import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


# Function to tune the model using GridSearchCV
def tune_model(model, param_grid, X_train, X_test, y_train, y_test):
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1) # Create a GridSearchCV object
    grid.fit(X_train, y_train) # Train the model
    y_pred = grid.predict(X_test) # Make predictions
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
    best_model = grid.best_estimator_ # Get the best model
    return accuracy, best_model


data_train = pd.read_csv('datasetTV.csv', header=None)
data_test = pd.read_csv('datasetTest.csv', header=None)

features = np.array(data_train)[:,:-1]
labels = np.array(data_train)[:,-1]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

MLP = MLPClassifier(random_state=1)

param_grid = {
    'hidden_layer_sizes': [(400,36)],
    'alpha': [1e-2,0.1,1,10]}
best_accuracy, best_model = tune_model(MLP, param_grid, X_train, X_test, y_train, y_test)

print(best_accuracy, best_model)

