import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Embedding, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import re
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('ML_Datasets/vehicle_data_with_weather.csv')

target = 'exit_edge'
categorical_columns = ['edge', 'lane', 'nextTLS', 'junction_id', 'exit_edge']
numeric_columns = ['spd', 'turnAngle', 'temperature', 'tl_phase_duration', 'tl_lanes_controlled', 'tflight']

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

# Features and target selection
features = [
    'spd', 'turnAngle', 'temperature', 'tl_phase_duration',
    'tl_lanes_controlled', 'tflight', 'edge', 'lane', 'nextTLS', 'junction_id'
]

X = df_cleaned[features].values
y = df_cleaned[target].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape input for Transformer, assuming tabular data doesn't need a sequence length
# We'll treat each row as a sequence of features, which can be embedded
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define Transformer model
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attention = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention)
    attention = Add()([attention, inputs])

    feed_forward = Dense(ff_dim, activation="relu")(attention)
    feed_forward = Dropout(dropout)(feed_forward)
    feed_forward = Dense(inputs.shape[-1])(feed_forward)
    outputs = LayerNormalization(epsilon=1e-6)(feed_forward)
    outputs = Add()([outputs, attention])
    return outputs

# Transformer model for tabular data
def build_transformer_model(input_shape, num_classes, head_size=128, num_heads=4, ff_dim=64, num_transformer_blocks=2, mlp_units=[128], dropout=0.1, mlp_dropout=0.1):
    inputs = Input(shape=input_shape)
    
    # Transformer blocks
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # MLP
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs)

# Build and compile the model
num_classes = len(np.unique(y))
model = build_transformer_model(input_shape=(1, X_train.shape[2]), num_classes=num_classes)

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2,
                    callbacks=[reduce_lr, early_stopping], verbose=1)

# Predictions and evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")

# Plotting the training and validation loss and accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Loss & Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

model.save('Models/exit_edge_transformer.h5')
