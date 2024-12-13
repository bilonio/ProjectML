import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd

# Generate example data (replace this with your dataset)
from sklearn.datasets import make_classification


data_train = pd.read_csv("datasetTV.csv", header=None)
data_test = pd.read_csv("datasetTest.csv", header=None)
X_data = data_train.iloc[:, :-1]
y_data = data_train.iloc[:, -1]
y_data = y_data - 1
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# Step 1: Train Base Models
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)

# Step 2: Generate Predictions from Base Models
# Base model predictions on the training set
rf_preds_train = rf.predict_proba(X_train)  # Use probabilities
xgb_preds_train = xgb.predict_proba(X_train)

# Base model predictions on the test set
rf_preds_test = rf.predict_proba(X_test)
xgb_preds_test = xgb.predict_proba(X_test)

# Step 3: Create Meta-Features
# Concatenate predictions from both models
meta_features_train = np.hstack((rf_preds_train, xgb_preds_train))
meta_features_test = np.hstack((rf_preds_test, xgb_preds_test))


def tune_model(model, param_grid):
    grid = GridSearchCV(model, param_grid, cv=3)  # Create a GridSearchCV object
    grid.fit(meta_features_train, y_train)  # Train the model
    y_pred = grid.predict(meta_features_test)  # Make predictions
    accuracy = grid.score(meta_features_test, y_test)  # Get the accuracy
    best_model = grid.best_estimator_  # Get the best model
    return accuracy, best_model


# Step 4: Train Meta-Model
meta_model = SVC()
param_grid = {
    "C": [10],
    "gamma": [0.01],
    "kernel": ["poly"],
    "class_weight": ["balanced"],
}
accuracy, meta_model = tune_model(meta_model, param_grid)

# Step 5: Make Final Predictions
# final_preds = meta_model.predict(meta_features_test)

# Evaluate Performance
# accuracy = accuracy_score(y_test, final_preds)
print(f"Stacked Model Accuracy: {accuracy:.4f}")
