import cv2
import numpy as np
import os

dataset_path = "dataset"
faces = []
labels = []
label_dict = {}

label = 0

# Check if dataset exists
if not os.path.exists(dataset_path):
    print("Dataset folder not found! Please run data_collection.py first.")
    exit()

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    # Skip if it's not a directory (skip files like dataset.py)
    if not os.path.isdir(person_path):
        print(f"Skipping {person} - not a directory")
        continue
    
    label_dict[label] = person
    print(f"Processing person: {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        # Only process image files
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Resize to consistent size
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(label)
            print(f"  Loaded: {img_name}")
        else:
            print(f"  Failed to load: {img_path}")

    label += 1

if len(faces) == 0:
    print("No faces found in dataset!")
    print("Make sure you have run data_collection.py first and collected face images.")
    exit()

faces = np.array(faces)
labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)
model.save("face_model.xml")

print(f"\n✅ Model trained successfully!")
print(f"📸 Total images: {len(faces)}")
print(f"👥 Total people: {label}")
print("📋 Labels mapping:", label_dict)