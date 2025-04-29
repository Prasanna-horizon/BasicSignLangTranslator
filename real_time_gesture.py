import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import mediapipe as mp
import pyttsx3  # Text-to-Speech
import threading  # Prevents lag from speech
import joblib  # For loading the scaler

# 🔹 Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()

def speak_text(text):
    threading.Thread(target=lambda: tts_engine.say(text) or tts_engine.runAndWait()).start()

# 🔹 Load the trained model
model_path = 'models/gesture_model.keras'
if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found! Exiting...")
    exit()

model = load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# 🔹 Load label encoder
csv_file = "gesture_data.csv"
df = pd.read_csv(csv_file)
label_encoder = LabelEncoder()
label_encoder.fit(df.iloc[:, 0].astype(str).values)

# 🔹 Load StandardScaler for normalization (if used in training)
scaler_path = "models/scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# 🔹 Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# 🔹 Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame from webcam.")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_gesture = None  # Default: no gesture detected

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # Flatten (x, y) pairs

            if len(landmarks) == 42:  # Ensure 21 landmarks (x, y)
                # Convert landmarks to NumPy array
                input_data = np.array(landmarks).reshape(1, -1)

                # Normalize landmarks if scaler exists
                if scaler:
                    input_data = scaler.transform(input_data)

                # 🔹 Predict gesture
                prediction = model.predict(input_data)
                predicted_class_index = np.argmax(prediction)
                predicted_gesture = label_encoder.inverse_transform([predicted_class_index])[0]

                # 🔹 Speak the recognized gesture
                speak_text(predicted_gesture)

                # 🔹 Display prediction on screen
                cv2.putText(frame, predicted_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # If no gesture was detected, display a message
    if not predicted_gesture:
        cv2.putText(frame, "No Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show webcam feed
    cv2.imshow('Real-Time Gesture Recognition', frame)

    # Reduce CPU usage slightly by adding a small delay
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
