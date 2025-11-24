# Emotion Detection Using Deep Learning and OpenCV

## Overview
This project is a real-time emotion detection system that uses a webcam to recognize facial expressions.  
A Convolutional Neural Network (CNN) trained on the FER-2013 dataset is used to classify emotions into seven categories:
- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

The system uses:
- TensorFlow/Keras for the deep learning model  
- OpenCV for real-time webcam processing and face detection  
- *Haar Cascade* for identifying faces in video frames  

---

##  Features
- Real-time face detection
- Fast and efficient emotion recognition
- Smooth webcam performance
- Seven emotion classes supported
- Easy-to-train and customizable model

---

## Model Architecture
The CNN model includes:
- 4 Convolutional layers (128, 256, 512, 512 filters)
- MaxPooling layers
- Dropout for regularization
- Fully connected dense layers
- Softmax output layer for 7-class classification

The model is saved as:
- emotiondetector.json (model architecture)
- emotiondetector.h5 (trained weights)

---

## Dataset
FER-2013 Dataset
- 48×48 grayscale images  
- 35,887 labeled samples  
- 7 emotion categories  

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Pandas  
- Haar Cascade Face Detection  

---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install tensorflow opencv-python numpy pandas tqdmscikit-learn
