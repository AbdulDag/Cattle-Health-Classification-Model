import os
import cv2
from ultralytics import YOLO

def auto_crop_dataset(input_dir, output_dir):
    # Load YOLO - Lowering confidence to 0.1 to catch cows in weird angles/lighting
    print("Loading YOLOv8...")
    cow_finder = YOLO('yolov8n.pt')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Note: Folder names are CASE SENSITIVE. Ensure they match your sidebar!
    for phase in ['train', 'val']:
        for condition in ['0_Healthy', '1_Sick']:
            current_input_folder = os.path.join(input_dir, phase, condition)
            current_output_folder = os.path.join(output_dir, phase, condition)

            if not os.path.exists(current_input_folder):
                print(f"Skipping: {current_input_folder} (Folder not found)")
                continue

            if not os.path.exists(current_output_folder):
                os.makedirs(current_output_folder)

            print(f"\n--- Entering {phase}/{condition} ---")
            
            files = os.listdir(current_input_folder)
            if not files:
                print(f"      [!] No files found in {current_input_folder}")

            for filename in files:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(current_input_folder, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"      [!] Error reading image: {filename}")
                    continue

                # Run YOLO with lower confidence and suppress terminal spam
                results = cow_finder(img, verbose=False, conf=0.5)
                
                cow_count = 0
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        
                        # Class 19 is Cow in the COCO dataset
                        if class_id == 19:
                            cow_count += 1
                            coords = box.xyxy[0]
                            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                            
                            # Slice the image [Y, X]
                            cow_crop = img[y1:y2, x1:x2]
                            
                            if cow_crop.size > 0:
                                new_filename = f"{os.path.splitext(filename)[0]}_cow{cow_count}.jpg"
                                save_path = os.path.join(current_output_folder, new_filename)
                                cv2.imwrite(save_path, cow_crop)
                                print(f"      [+] Saved {new_filename} (Conf: {conf_score:.2f})")
                        else:
                            # Useful for debugging if it's misclassifying cows as dogs/horses
                            print(f"      [?] Saw Class {class_id}, ignoring.")
                
                if cow_count == 0:
                    print(f"      [!] YOLO found NO COWS in {filename}")

if __name__ == "__main__":
    # Ensure these names match the folders in your 'dataset_split' directory
    INPUT_FOLDER = "dataset_split" 
    OUTPUT_FOLDER = "dataset_cropped" 
    
    print("Starting the Automated Dataset Cropper...")
    auto_crop_dataset(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\nCheck your 'dataset_cropped' folder now!")