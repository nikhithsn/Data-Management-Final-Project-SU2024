#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 00:52:10 2024

@author: nikhith
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

pd.set_option('max_columns', None)


# DATA COLLECTION

# Load dataset
path = 'HDHI Admission data.csv'

#data = pd.read_csv(path, encoding='latin-1', error_bad_lines=False, lineterminator='\n')
data = pd.read_csv(path)

print(data.head())



# DATA CLEANING

data.drop(columns = ['D.O.A', 'D.O.D'], inplace  = True)

# Changes selected columns froms from string identifiers to binary identifiers
    # M = 1, F = 0
    # Rural = 0, Urban = 1
    # Emergency = 1, Outpatient = 0

le = LabelEncoder()
#data['GENDER'] = le.fit_transform(data['GENDER'])
#data['RURAL'] = le.fit_transform(data['RURAL'])
#data['TYPE OF ADMISSION-EMERGENCY/OPD'] = le.fit_transform(data['TYPE OF ADMISSION-EMERGENCY/OPD'])

# Fixing CSV Format Issues
data = data.rename(columns={'SMOKING ': 'SMOKING'})

# Removing Duplicates
data.drop_duplicates(inplace=True)

print(data.head())

data.columns



# DATA STORAGE
    
# Database connection
engine = create_engine('mysql+pymysql://root:01xmrmdvxe@127.0.0.1:3306/data_management')

# Load the dataset into MySQL
data.to_sql('hospital_data', con=engine, if_exists='replace', index=False)

# Handling Missing Values
data.fillna(method='ffill', inplace=True)



# DATA TRANSFORMATION

# Feature Engineering
data['age_group'] = pd.cut(data['AGE'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])

# Normalization and Scaling
scaler = StandardScaler()
data[['AGE', 'DURATION OF STAY']] = scaler.fit_transform(data[['AGE', 'DURATION OF STAY']])

# Display transformed data
print(data.head())



# EXPLORATORY DATA ANALYSIS

# Visualize age groups with a specific disease
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='age_group', hue='disease')
plt.title('Age Group Distribution by Disease')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()



# PREDICTIVE MODELING

# Prepare data for modeling
X = data[['age', 'length_of_stay']]  # Features
y = data['disease']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')



# MODEL DEPLOYMENT

# Save the model
joblib.dump(model, 'model.pkl')

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)