
# 🧠 Sign Language Translator Using Computer Vision & Machine Learning

This project is a **real-time Sign Language Translator** that uses a webcam to recognize hand gestures and convert them into text and speech. It utilizes **Computer Vision**, **Machine Learning**, and **Deep Learning** techniques to interpret sign language effectively.

---

## 💡 Features

- Real-time gesture recognition using webcam input  
- Translates hand gestures into corresponding text  
- Converts translated text into speech  
- Preprocessing and landmark extraction using **MediaPipe**
- Model training using **scikit-learn** and **Keras**
- Modular code for easy extension

---

## 🛠️ Project Modules

- `image_to_csv.py` – Capture hand landmarks and export to CSV  
- `data_preprocessing.py` – Normalize and preprocess dataset  
- `model_training.py` – Train gesture recognition model  
- `predict_gesture.py` – Load model and predict gestures  
- `real_time_gesture.py` – Real-time webcam detection and translation  

---

## ✅ Requirements to Run

Ensure the following are installed:

- **Python 3.9.12**
- Libraries:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `keras`
  - `pyttsx3` *(for speech output)*

You can install dependencies with:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn keras pyttsx3
```

---

## 🚀 How to Run

1. **Capture landmarks**  
   Run `image_to_csv.py` to collect gesture data (if required)

2. **Preprocess data**  
   Run `data_preprocessing.py` to prepare data for training

3. **Train the model**  
   Run `model_training.py` to train and save your model

4. **Predict single image**  
   Use `predict_gesture.py` to predict a gesture from static image/landmarks

5. **Real-time recognition**  
   Launch `real_time_gesture.py` to start webcam detection and translation

---

## ⚠️ Changes Needed In Order To Run The Code

- **In image_to_csv.py, change the path to point to your dataset image directory before running the script.**
  Example:
  DATASET_DIR = 'path//to//your//dataset'
