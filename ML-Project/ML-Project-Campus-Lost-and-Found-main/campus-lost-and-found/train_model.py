# File: train_model.py
# Description: specific script to train the Random Forest Classifier.
# Changes: Refactored logic flow, updated hyperparameters, and variable renaming.

import os
import numpy as np
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from modules import features

# --- SYSTEM CONSTANTS ---
RAW_DATA_DIR = "data/raw_dataset" 
MODEL_OUTPUT_FILE = "modules/category_classifier.pkl"

def compile_feature_set():
    """
    Traverses the dataset directory, extracts visual features (HOG + Color),
    and aggregates them into numpy arrays.
    """
    print(">>> Phase 1: Initiating Data Ingestion & Feature Extraction...")
    
    features_list = [] 
    labels_list = [] 
    
    # Validation check
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[CRITICAL] Directory not found: {RAW_DATA_DIR}")
        return None, None

    img_counter = 0
    
    # Recursive directory walk
    for dir_path, _, filenames in os.walk(RAW_DATA_DIR):
        for fname in filenames:
            # Filter for valid image formats
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(dir_path, fname)
                
                # Extract the class label from the parent folder name
                class_label = os.path.basename(dir_path)
                
                try:
                    # Invoke feature extractors from the modules package
                    vec_color = features.get_raw_color_hist(full_path)
                    vec_hog = features.get_hog_features(full_path)
                    
                    # Ensure both extractors returned valid data
                    if vec_color is not None and vec_hog is not None:
                        # Stack features horizontally
                        combined_vector = np.hstack((vec_color, vec_hog))
                        
                        features_list.append(combined_vector)
                        labels_list.append(class_label)
                        
                        img_counter += 1
                        # Log progress every 200 images
                        if img_counter % 200 == 0:
                            print(f"    -> Extracted features for {img_counter} items...")
                        
                except Exception as err:
                    # Silent fail on individual corrupt images to keep process running
                    continue
                
    print(f"[COMPLETED] Total samples ready: {len(features_list)}")
    return np.array(features_list), np.array(labels_list)

def execute_pipeline():
    """
    Loads data, splits into subsets, trains the Random Forest, 
    and serializes the model to disk.
    """
    # 1. Get Data
    X_data, y_data = compile_feature_set()
    
    if X_data is None or len(X_data) == 0:
        print("[ABORT] Dataset is empty or failed to load.")
        return

    print(f">>> Phase 2: Training Classifier on {len(X_data)} inputs...")
    
    # 2. Split Data (Using 25% for testing instead of 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.25, random_state=101
    )
    
    # 3. Initialize Model 
    # Increased estimators to 200 for robustness
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        n_jobs=-1, 
        random_state=101
    )
    
    # 4. Fit Model
    rf_model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = rf_model.predict(X_val)
    current_acc = accuracy_score(y_val, predictions)
    print(f"    -> Model Validation Accuracy: {current_acc*100:.2f}%")

    # 6. Serialize (Save)
    print(">>> Phase 3: Exporting Model...")
    try:
        with open(MODEL_OUTPUT_FILE, "wb") as file_out:
            pkl.dump(rf_model, file_out)
        print(f"[DONE] Classifier successfully saved to: {MODEL_OUTPUT_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")

if __name__ == "__main__":
    execute_pipeline()