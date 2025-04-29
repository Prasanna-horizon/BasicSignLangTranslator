# Sign Language Translator 🤟

This project is a real-time Sign Language Translator that detects and classifies hand gestures from sign language alphabets using computer vision and machine learning techniques.

## 📌 Features

- Real-time hand gesture detection using webcam
- Landmark detection with MediaPipe
- Trained model for classification of sign language letters (limited set)
- CSV data generation from hand landmarks
- Modular code structure for preprocessing, training, prediction, and real-time translation

## 📁 Project Structure

```
├── image_to_csv.py          # Generate landmark data from hand images
├── data_preprocessing.py    # Clean and preprocess the landmark CSV data
├── model_training.py        # Train and save a gesture recognition model
├── predict_gesture.py       # Predict gesture from an image using the model
├── real_time_gesture.py     # Real-time detection using webcam feed
└── dataset/                 # Folder to store hand gesture image data
```

## ⚠️ Limitations

- The current model is trained on a **limited subset of letters** only (e.g., A–E or A–J)
- Real-time accuracy may vary depending on lighting and hand positioning
- No full sentence recognition — this is alphabet/gesture-only

## 💻 Requirements to Run

- Python 3.9.12
- Libraries:
  - OpenCV
  - MediaPipe
  - NumPy
  - Pandas
  - scikit-learn
  - Keras

You can install them using:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn keras
```

## 🚀 How to Run

1. Clone this repository.
2. Run `image_to_csv.py` to generate the data CSV.
3. Run `data_preprocessing.py` to prepare the dataset.
4. Run `model_training.py` to train the gesture model.
5. Run `real_time_gesture.py` to start real-time detection using webcam.