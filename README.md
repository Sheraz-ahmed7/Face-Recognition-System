Face Recognition System 👨‍💼🔍
A real-time face recognition system built with Python and OpenCV that can detect, collect, and recognize multiple faces using LBPH (Local Binary Patterns Histograms) algorithm.

📋 Overview
This project implements a complete face recognition pipeline:

Face Detection using Haar Cascade Classifier

Data Collection for multiple users

Model Training with LBPH algorithm

Real-time Face Recognition through webcam

✨ Features
✅ Real-time face detection from webcam feed

✅ Multi-user support - train with multiple people

✅ Automatic dataset creation - organized by person name

✅ LBPH model training for accurate recognition

✅ Confidence-based recognition with "Unknown" detection

✅ Simple keyboard controls (Enter key to exit)

🛠️ Technology Stack
Python 3.6+

OpenCV (with contrib modules)

NumPy

Haar Cascade Classifier for face detection

LBPH Face Recognizer for face recognition

📁 Project Structure

face-recognition-system/
│
├── face_detection.py      # Basic face detection test
├── data_collection.py     # Collect face samples for training
├── train.py               # Train the recognition model
├── recognition.py         # Real-time face recognition
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
│
├── dataset/               # Created automatically
│   ├── person1/          # 50-150 face images per person
│   └── person2/
│
└── face_model.xml         # Trained model (generated after training)


🔧 Installation

Prerequisites
Python 3.6 or higher
Webcam
pip package manager

🚀 Usage Guide
Step 1: Basic Face Detection Test
python face_detection.py

Tests if your webcam and face detection are working properly.

Step 2: Collect Training Data
python data_collection.py

Enter the person's name when prompted

The system will capture 50-150 face images

Move your face slightly for variations

Press Enter or wait for count to finish

Step 3: Train the Model
python train.py
Reads all collected images from dataset/ folder

Trains the LBPH recognizer

Saves model as face_model.xml

python recognition.py

Webcam opens with real-time face recognition

Recognized faces show names with green boxes

Unknown faces show "Unknown" with red boxes

Press Enter to exit

📊 How It Works
Face Detection
Uses Haar Cascade Classifier to detect faces in grayscale images:
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

Data Collection
Creates person-specific folders in dataset/

Saves 50-150 grayscale face images per person

Ensures diverse angles and expressions for better training

Model Training
Uses LBPH (Local Binary Patterns Histograms) algorithm

Converts faces to uniform size (200x200)

Creates a histogram model for each person

Saves trained model to XML file

Recognition
Compares live face with trained models

Returns confidence score (lower = better match)

Threshold of <70 for positive identification

Shows "Unknown" for low-confidence matches

⚙️ Configuration
Adjust these parameters for better performance:

In data_collection.py:
count == 150  # Increase from 50 to 150 for more samples

In recognition.py:
if confidence < 50:  # Adjust threshold (lower = stricter)
    # Recognized
else:
    # Unknown

🎯 Performance Tips
Lighting: Ensure good, even lighting

Distance: Stay 1-3 feet from camera

Variations: Collect images with different expressions and angles

Quantity: More samples (100-150) per person = better accuracy

Background: Plain background helps detection

Accessories: Train with/without glasses if applicable
