import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import traci

# Load Data
data = pd.read_csv('ML_Datasets/vehicle_data_with_weather.csv')

# Feature selection and preprocessing
features = data[['spd', 'temperature', 'tl_state', 'displacement']]  # Add relevant features
features.loc[:, 'tl_state'] = features['tl_state'].map({'R': 0, 'G': 1, 'Y': 2})

# Check for NaN values and handle them
print(features.isnull().sum())
features.fillna(0, inplace=True)  # Handle NaN values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Define labels
labels = pd.to_numeric(data['exit_edge'], errors='coerce')
labels.fillna(0, inplace=True)  # Handle NaN labels

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

# Define the Model
model = Sequential()
model.add(input(shape=(X_train.shape[1],)))  # Use Input layer for defining input shape
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))  # Adjust activation based on output

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
