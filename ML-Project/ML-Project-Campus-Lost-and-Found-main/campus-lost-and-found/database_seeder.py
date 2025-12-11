# File: database_seeder.py
# Description: Automated script to ingest raw images, generate metadata, 
#              and populate the SQL database for testing.
# Updated: Refactored for modularity and distinct logic flow.

import os
import shutil
import random
from modules import db, features, auth

# --- PATH CONFIGURATION ---
RAW_INPUT_DIR = "data/raw_dataset"
FINAL_IMG_DIR = "data/uploaded_images"

# --- DATASET DEFINITIONS ---
# List of recognized item classes
TARGET_CLASSES = [
    "Backpack", "Bracelet", "Calculator", "Charger", "Earphones", 
    "Headphones", "Keyboard", "Keys", "Laptop", "Mouse", 
    "Smartphone", "Waterbottle", "Wristwatch", "Other"
]

# Attribute pools for metadata generation
ATTR_PALETTE = ["Blue", "Red", "Matte Black", "White", "Silver", "Neon Green", "Yellow", "Grey"]
ATTR_SPOTS = ["Library 2nd Floor", "Gym Lockers", "Main Cafeteria", "Student Union", "Parking Garage B", "Lecture Hall A", "Corridor"]

def get_or_create_admin():
    """Ensures a bot/admin account exists to own the seeded items."""
    bot_name = "DataSeederBot"
    
    # Check if user exists
    user_record = db.get_user_by_username(bot_name)
    
    if user_record:
        return user_record['id']
    else:
        # Register new admin if missing
        print(f"[*] Registering new admin: {bot_name}")
        return auth.register_user(bot_name, "bot_secret_pass", "bot@admin.com")

def generate_description(category):
    """Creates a varied description string."""
    color = random.choice(ATTR_PALETTE)
    place = random.choice(ATTR_SPOTS)
    return f"{category} detected. Color: {color}. Last seen area: {place}."

def process_artifact(file_path, filename, root_folder, admin_id):
    """Handles the processing of a single image file."""
    try:
        # 1. Infer Category from folder name
        folder_name = os.path.basename(root_folder)
        detected_label = "Other"
        
        for label in TARGET_CLASSES:
            if label.lower() in folder_name.lower():
                detected_label = label
                break
        
        # 2. Prepare Destination
        new_filename = f"auto_{folder_name}_{filename}"
        target_path = os.path.join(FINAL_IMG_DIR, new_filename)
        
        # 3. Copy File
        shutil.copy(file_path, target_path)
        
        # 4. Generate Metadata & Vectors
        desc_text = generate_description(detected_label)
        
        # Calls to ML modules
        vec_visual = features.extract_visual_vector(target_path)
        vec_text = features.extract_text_vector(desc_text)
        
        # 5. Commit to DB
        if vec_visual is not None:
            db.add_item(
                user_id=admin_id,
                item_type="FOUND",
                category=detected_label,
                description=desc_text,
                image_path=target_path,
                features_col=vec_visual,
                features_txt=vec_text
            )
            return True, detected_label
        return False, "Vector Extraction Failed"

    except Exception as ex:
        return False, str(ex)

def populate_database():
    print(">>> INITIALIZING DATA INGESTION <<<")
    
    # Setup DB
    db.init_db()
    
    # Pre-flight checks
    if not os.path.exists(RAW_INPUT_DIR):
        print(f"[!] Critical: Source directory '{RAW_INPUT_DIR}' missing.")
        return

    if not os.path.exists(FINAL_IMG_DIR):
        os.makedirs(FINAL_IMG_DIR)

    # Get Admin ID
    admin_id = get_or_create_admin()
    
    total_indexed = 0
    
    # Recursive Walk
    for root, _, files in os.walk(RAW_INPUT_DIR):
        for f_name in files:
            # Filter valid image extensions
            if f_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_source_path = os.path.join(root, f_name)
                
                success, msg = process_artifact(full_source_path, f_name, root, admin_id)
                
                if success:
                    print(f"   [OK] Indexed: {msg}")
                    total_indexed += 1
                else:
                    print(f"   [ERR] Failed {f_name}: {msg}")

    print(f"\n>>> PROCESS COMPLETE. Total Items Seeded: {total_indexed} <<<")

if __name__ == "__main__":
    populate_database()