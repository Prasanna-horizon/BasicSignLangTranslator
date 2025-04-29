import os
import joblib
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load StandardScaler
scaler_path = "models/scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Extract landmarks
def extract_landmarks(image):
    """Extracts 21 (x, y) hand landmarks from an image using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y])

        return np.array(landmarks) if len(landmarks) == 42 else None

    return None

# Data augmentation
def augment_image(image):
    """Applies random transformations (flipping, rotation) to increase dataset size."""
    aug_images = [image]  # Original image
    aug_images.append(cv2.flip(image, 1))  # Flip horizontally
    aug_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))  # Rotate 90 degrees
    return aug_images

# Preprocess image
def preprocess_image(image_path, augment=False):
    """Loads, augments, extracts landmarks, and scales data for training/prediction."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Cannot read image {image_path}. Skipping.")
        return None

    images = augment_image(image) if augment else [image]
    processed_landmarks = []

    for img in images:
        landmarks = extract_landmarks(img)
        if landmarks is not None:
            if scaler:  # Normalize data if scaler is available
                landmarks = scaler.transform([landmarks])[0]
            processed_landmarks.append(landmarks)

    return processed_landmarks if processed_landmarks else None
