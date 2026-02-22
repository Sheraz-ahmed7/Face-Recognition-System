import cv2
import numpy as np
import os
from collections import deque

def preprocess_face(face):
    """Apply preprocessing for better recognition"""
    face = cv2.resize(face, (200, 200))
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    return face

def get_adaptive_threshold(frame, base_threshold=70):
    """Calculate adaptive threshold based on lighting"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Adjust threshold based on brightness
    # Darker scenes -> lower threshold (easier to recognize)
    # Brighter scenes -> higher threshold (stricter)
    brightness_factor = (avg_brightness - 127) / 127  # -1 to 1
    adjusted_threshold = base_threshold - brightness_factor * 20
    
    return max(50, min(90, adjusted_threshold))  # Clamp between 50-90

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load model
try:
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("face_model.xml")
    print("✅ Model loaded successfully!")
except:
    print("❌ Model not found! Please train first.")
    exit()

# Load names
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    print("❌ Dataset folder not found!")
    exit()

names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
print(f"👥 Recognizing {len(names)} people: {', '.join(names)}")

# For temporal smoothing (reduce flickering)
face_history = deque(maxlen=5)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Starting recognition... Press 'q' to quit, 's' to save screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get adaptive threshold for this frame
    adaptive_threshold = get_adaptive_threshold(frame)
    
    # Detect faces with multiple scales
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Smaller step for better detection
        minNeighbors=5,
        minSize=(60, 60),     # Minimum face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    current_faces = []
    
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_roi = gray[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        
        # Predict
        label, confidence = model.predict(processed_face)
        
        # Apply temporal smoothing
        current_faces.append((x, y, w, h, label, confidence))
        
        # Determine name and color based on confidence
        if confidence < adaptive_threshold and 0 <= label < len(names):
            name = names[label]
            color = (0, 255, 0)  # Green for recognized
            # Add confidence to display
            conf_text = f"{confidence:.1f}"
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            conf_text = f"{confidence:.1f}"
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw name with background
        label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
        cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Draw confidence
        cv2.putText(frame, f"conf: {conf_text}", (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Display info
    cv2.putText(frame, f"Threshold: {adaptive_threshold:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow("Face Recognition System", frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 13:  # 'q' or Enter
        break
    elif key == ord('s'):  # 's' to save screenshot
        cv2.imwrite("recognition_screenshot.jpg", frame)
        print("📸 Screenshot saved!")

cap.release()
cv2.destroyAllWindows()
print("👋 Recognition stopped")