import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import re
import matplotlib.pyplot as plt

target = 'exit_edge'
categorical_columns = ['edge', 'lane', 'nextTLS', 'junction_id', 'exit_edge']
numeric_columns = ['spd', 'turnAngle', 'temperature', 'tl_phase_duration', 'tl_lanes_controlled', 'tflight']

df = pd.read_csv('ML_Datasets/vehicle_data_with_weather.csv')

# Cleaning edge values
def clean_edge_values(edge_str):
    if isinstance(edge_str, str):
        return re.split(r"[#()]", edge_str)[0] 
    return np.nan 

df['exit_edge'] = df['exit_edge'].apply(clean_edge_values)
df['edge'] = df['edge'].apply(clean_edge_values)
df['lane'] = df['lane'].apply(clean_edge_values)

# Label encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].fillna('missing')) 

# Convert numeric columns to numeric and handle missing values
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    if df[col].notna().any(): 
        df[col].fillna(df[col].median(), inplace=True) 
    else:
        df[col].fillna(0, inplace=True) 

# Transform 'exit_edge' into a categorical label (classification task)
df['exit_edge'] = label_encoder.fit_transform(df['exit_edge'].fillna('missing'))

# Ensure no missing values after cleaning
df_cleaned = df.dropna(subset=['exit_edge'])

if df_cleaned.empty:
    print("No valid rows left after cleaning.")
else:
    # Features and target selection
    features = [
        'spd', 'turnAngle', 'temperature', 'tl_phase_duration',
        'tl_lanes_controlled', 'tflight', 'edge', 'lane', 'nextTLS', 'junction_id'
    ]
    
    X = df_cleaned[features].values
    y = df_cleaned[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get number of unique classes for classification
    num_classes = len(np.unique(y))

    # Model definition
    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # Classification layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Callbacks for learning rate reduction and early stopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2,
                        callbacks=[reduce_lr, early_stopping], verbose=1)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.save('Models/exit_edge_sequential.h5')
