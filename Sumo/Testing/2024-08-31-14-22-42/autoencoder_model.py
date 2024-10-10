import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import re
import matplotlib.pyplot as plt

target = 'exit_edge'
categorical_columns = ['edge', 'lane', 'nextTLS', 'junction_id', 'exit_edge']
numeric_columns = ['spd', 'turnAngle', 'temperature', 'tl_phase_duration', 'tl_lanes_controlled', 'tflight']

df = pd.read_csv('ML_Datasets/vehicle_data_with_weather.csv')

def clean_edge_values(edge_str):
    if isinstance(edge_str, str):
        return re.split(r"[#()]", edge_str)[0] 
    return np.nan 

df['exit_edge'] = df['exit_edge'].apply(clean_edge_values)
df['edge'] = df['edge'].apply(clean_edge_values)
df['lane'] = df['lane'].apply(clean_edge_values)

label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].fillna('missing')) 

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    if df[col].notna().any(): 
        df[col].fillna(df[col].median(), inplace=True) 
    else:
        df[col].fillna(0, inplace=True) 

df['exit_edge'] = label_encoder.fit_transform(df['exit_edge'].fillna('missing'))

df_cleaned = df.dropna(subset=['exit_edge'])

if df_cleaned.empty:
    print("No valid rows left after cleaning.")
else:
    features = [
        'spd', 'turnAngle', 'temperature', 'tl_phase_duration',
        'tl_lanes_controlled', 'tflight', 'edge', 'lane', 'nextTLS', 'junction_id'
    ]
    
    X = df_cleaned[features].values
    y = df_cleaned[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    encoding_dim = 8

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded_output = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    encoder = Model(inputs=input_layer, outputs=encoded_output)

    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    num_classes = len(np.unique(y))

    model = Sequential()

    model.add(Input(shape=(X_train_encoded.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(X_train_encoded, y_train, epochs=15, batch_size=32, validation_split=0.2,
                    callbacks=[reduce_lr, early_stopping], verbose=1) 

    y_pred = np.argmax(model.predict(X_test_encoded), axis=1)

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

    model.save('Models/exit_edge_autoencoder.h5')
