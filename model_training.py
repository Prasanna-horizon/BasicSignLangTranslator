import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Load dataset
csv_file = "gesture_data.csv"
df = pd.read_csv(csv_file)

# Check class distribution
class_counts = df["gesture"].value_counts()
print("📊 Class Distribution:\n", class_counts)

# Warn if imbalance is found
if class_counts.min() < 10:
    print("⚠ Warning: Some gestures have fewer than 10 images. Consider collecting more data.")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df.iloc[:, 0].values)
X = df.iloc[:, 1:].values  # Extract features (landmarks)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalize landmarks (Standardization improves model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future predictions
import joblib
joblib.dump(scaler, "models/scaler.pkl")

# Build Model
model = models.Sequential([
    layers.InputLayer(input_shape=(42,)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(len(np.unique(y)), activation='softmax')  # Auto-detects classes
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
history = model.fit(X_train, y_train, 
                    epochs=50, batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping])

# Save Model & Label Encoder
model.save("models/gesture_model.keras")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("✅ Model & Label Encoder saved!")

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🎯 Final Test Accuracy: {test_acc:.2%}")
