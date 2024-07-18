#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

data = pd.read_csv(path)

print(data.head())



# DATA CLEANING

data.drop(columns = ['D.O.A', 'D.O.D'], inplace  = True)

le = LabelEncoder()

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

#Grouping by age:
data['age_group'] = pd.cut(data['AGE'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])

#Grouping by Diabetes
data['GLUCOSE'] = data['GLUCOSE'].replace('EMPTY', -1)
data['GLUCOSE'] = data.GLUCOSE.astype(float)
data['diabetic_classification'] = pd.cut(data['GLUCOSE'], bins=[-10, -1, 139, 190, 1000], labels=['Undetermined', 'Not Diabetic', 'Pre-Diabetic', 'Diabetic'])


# Display transformed data
print(data.head())
data


# EXPLORATORY DATA ANALYSIS

# Visualize age groups with a specific disease
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='age_group', hue='GLUCOSE')
plt.title('Age Group Distribution by Glucose Levels')
plt.xlabel('Age Group')
plt.ylabel('Glucose Level')
plt.legend().set_visible(False)
plt.show()

#Visualize heart failure across age groups
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='age_group', hue='HEART FAILURE')
plt.title('Age Group Distribution by Heart Failure')
plt.xlabel('Age Group')
plt.ylabel('Number of Patients with Heart Failure')
plt.show()

#Visualize diabetes across age groups
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='age_group', hue='diabetic_classification')
plt.title('Diabetes Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of Patients')
plt.show()



# PREDICTIVE MODELING

# Preparing data for modeling
data['HB'] = data['HB'].replace('EMPTY', -1)
data['CREATININE'] = data['CREATININE'].replace('EMPTY', -1)

X = data[['AGE', 'HB', 'CREATININE']]  # Features
y = data['diabetic_classification']  # Target variable

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Model evaluatiion
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')



# MODEL DEPLOYMENT

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
