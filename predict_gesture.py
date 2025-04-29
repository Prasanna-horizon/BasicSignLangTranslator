import os
import numpy as np
import cv2
import mediapipe as mp
import joblib

# Load StandardScaler
scaler_path = "models/scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands

# Extract landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y])
            
            return np.array(landmarks) if len(landmarks) == 42 else None
    
    return None

# Normalize landmarks if scaler exists
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Cannot read image {image_path}. Skipping.")
        return None
    
    landmarks = extract_landmarks(image)
    if landmarks is None:
        return None
    
    # Apply normalization if scaler is available
    if scaler:
        landmarks = scaler.transform([landmarks])[0]
    
    return landmarks
