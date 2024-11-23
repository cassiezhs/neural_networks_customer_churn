# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           classification_report, log_loss, confusion_matrix,
                           roc_curve, precision_recall_curve, accuracy_score, 
                            confusion_matrix, roc_auc_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# load the data
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv",
                   low_memory=False)

# handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna(data['TotalCharges'].mean(), inplace=True)

# convert categorical variables to numerical
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'customerID':  # Exclude customerID
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# split data into features and target
X = data.drop(['Churn', 'customerID'], axis=1)
y = data['Churn']

# split data into train vs test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural network model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# evaluate the Nerual Network model
nn_y_pred_proba = nn_model.predict(X_test)
nn_y_pred = (nn_y_pred_proba > 0.5).astype(int)
nn_accuracy = accuracy_score(y_test, nn_y_pred)
print("\nNeural Network Results:")
print(f"Accuracy: {nn_accuracy:.2f}")
print(classification_report(y_test, nn_y_pred))

# Train a logistic regression model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Make predictions
log_y_pred = log_model.predict(X_test)
log_y_pred_proba = log_model.predict_proba(X_test)[:, 1]

# Evaluate the model
log_accuracy = accuracy_score(y_test, log_y_pred)
log_roc_auc = roc_auc_score(y_test, log_y_pred_proba)
log_conf_matrix = confusion_matrix(y_test, log_y_pred)
log_classification_rep = classification_report(y_test, log_y_pred)

# Display results
print("Logistic Regression Results:")
print(f"Accuracy: {log_accuracy:.2f}")
print(f"ROC-AUC: {log_roc_auc:.2f}")
print("\nConfusion Matrix:")
print(log_conf_matrix)
print("\nClassification Report:")
print(log_classification_rep)

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Making predictions on the test set
dt_y_pred = clf.predict(X_test)

# Evaluating the model
# Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f"Accuracy: {dt_accuracy:.2%}")

# Generate the confusion matrix
dt_conf_matrix = confusion_matrix(y_test, dt_y_pred)
print("Confusion Matrix:")
print(dt_conf_matrix)

# Generate the classification report
dt_class_report = classification_report(y_test, dt_y_pred)
print("Classification Report:")
print(dt_class_report)

# Building and evaluating Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(classification_report(y_test, rf_y_pred))

# Define a function to evaluate models
def evaluate_model(y_test, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Evaluate all models
evaluate_model(y_test, nn_y_pred, "Neural Network")
evaluate_model(y_test, log_y_pred, "Logistic Regression")
evaluate_model(y_test, dt_y_pred, "Decision Tree")
evaluate_model(y_test, rf_y_pred, "Random Forest")   