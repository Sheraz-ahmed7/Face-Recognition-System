import cv2
import numpy as np
import os

def preprocess_face(face):
    """Apply preprocessing for better recognition"""
    face = cv2.resize(face, (200, 200))
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    return face

def augment_face(face):
    """Create variations of face for better training"""
    augmented = [face]  # Original
    
    # Rotated ±5 degrees
    rows, cols = face.shape
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(face, M, (cols, cols))
        augmented.append(rotated)
    
    # Flipped
    augmented.append(cv2.flip(face, 1))
    
    # Brightness variations
    augmented.append(cv2.convertScaleAbs(face, alpha=1.2, beta=20))
    augmented.append(cv2.convertScaleAbs(face, alpha=0.8, beta=-20))
    
    return augmented

dataset_path = "dataset"
faces = []
labels = []
label_dict = {}

print("🔄 Starting training pipeline...")
label = 0

# Check if dataset exists
if not os.path.exists(dataset_path):
    print("❌ Dataset folder not found! Please run data_collection.py first.")
    exit()

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    if not os.path.isdir(person_path):
        print(f"⏭️ Skipping {person} - not a directory")
        continue
    
    label_dict[label] = person
    print(f"\n👤 Processing person: {person}")
    
    person_faces = 0
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Preprocess face
            processed_face = preprocess_face(img)
            
            # Augment and add to training set
            augmented_faces = augment_face(processed_face)
            for aug_face in augmented_faces:
                faces.append(aug_face)
                labels.append(label)
                person_faces += 1
        else:
            print(f"  ⚠️ Failed to load: {img_name}")

    print(f"  ✅ Added {person_faces} training samples for {person}")
    label += 1

if len(faces) == 0:
    print("❌ No faces found in dataset!")
    exit()

faces = np.array(faces)
labels = np.array(labels)

print(f"\n📊 Training with {len(faces)} total samples for {label} people")

# Train model
model = cv2.face.LBPHFaceRecognizer_create(
    radius=2,           # LBP radius
    neighbors=8,        # Number of neighbors
    grid_x=8,           # Grid cells in X direction
    grid_y=8,           # Grid cells in Y direction
    threshold=100.0     # Default confidence threshold
)

model.train(faces, labels)
model.save("face_model.xml")

print(f"\n✅ Model trained successfully!")
print(f"📸 Total training samples: {len(faces)}")
print(f"👥 People in database: {label}")
print("📋 Label mapping:")
for idx, name in label_dict.items():
    print(f"   {idx}: {name}")