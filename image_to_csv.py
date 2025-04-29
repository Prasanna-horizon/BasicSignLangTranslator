import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

# Initialize MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Set dataset path
image_folder = "C:\\Folder\\Path\\To\\Dataset_img"
csv_file = "gesture_data.csv"

# Define column headers
columns = ["gesture"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
df = pd.DataFrame(columns=columns)

# Limit max samples per gesture (to balance dataset)
MAX_SAMPLES = 500  # Prevents overfitting on "2" and "T"

# Function to augment images (flip, rotate)
def augment_image(image):
    flipped = cv2.flip(image, 1)  # Flip horizontally
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees
    return [image, flipped, rotated]

# Extract hand landmarks
def extract_landmarks(image):
    image = cv2.resize(image, (256, 256))  # Resize for speed
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    if result.multi_hand_landmarks:
        landmarks = [(lm.x, lm.y) for lm in result.multi_hand_landmarks[0].landmark]
        if len(landmarks) == 21:
            return [coord for x, y in landmarks for coord in (x, y)]  # Flatten (x,y)

    return None

# Process dataset
gesture_counts = {}

for gesture_folder in os.listdir(image_folder):
    gesture_folder_path = os.path.join(image_folder, gesture_folder)

    if os.path.isdir(gesture_folder_path):
        count = 0
        samples = []  # Store samples before adding to DataFrame

        for filename in os.listdir(gesture_folder_path):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(gesture_folder_path, filename)
                img = cv2.imread(image_path)

                # Process original + augmented images
                for aug_img in augment_image(img):
                    landmarks = extract_landmarks(aug_img)
                    if landmarks:
                        samples.append([gesture_folder] + landmarks)
                        count += 1
                        if count >= MAX_SAMPLES:  # Stop when max samples reached
                            break
            
            if count >= MAX_SAMPLES:
                break  # Stop processing more images

        # Save balanced samples (Only if there are valid samples)
        if samples:  
            df = pd.concat([df, pd.DataFrame(samples, columns=columns)], ignore_index=True)
        gesture_counts[gesture_folder] = min(count, MAX_SAMPLES)

# Save final CSV
df.to_csv(csv_file, index=False)
print("✅ CSV generation complete!")

# Show dataset balance
print("📊 Gesture Data Distribution:")
print(pd.Series(gesture_counts))
