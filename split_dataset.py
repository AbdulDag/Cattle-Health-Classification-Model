import os
import random
import shutil

def create_split(source_dir, train_dir, val_dir, split_ratio=0.7):
    # Ensure the target directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all the files in the source folder
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Shuffle the files randomly
    random.shuffle(files)
    
    # Calculate exactly where to make the cut
    split_index = int(len(files) * split_ratio)
    
    train_files = files[:split_index]
    val_files = files[split_index:]

    print(f"Splitting '{os.path.basename(source_dir)}': {len(train_files)} for Training, {len(val_files)} for Validation.")

    # Copy files to the Training folder
    for f in train_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(train_dir, f))
        
    # Copy files to the Validation folder
    for f in val_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(val_dir, f))


if __name__ == "__main__":
    base_source = "cow_dataset"
    sick_source = os.path.join(base_source, "1_Sick")
    healthy_source = os.path.join(base_source, "0_Healthy")

    base_dest = "dataset_split"
    
    train_sick = os.path.join(base_dest, "train", "1_Sick")
    train_healthy = os.path.join(base_dest, "train", "0_Healthy")
    
    val_sick = os.path.join(base_dest, "val", "1_Sick")
    val_healthy = os.path.join(base_dest, "val", "0_Healthy")

    print("Starting the 70-30 Stratified Split...")

    create_split(sick_source, train_sick, val_sick, split_ratio=0.7)
    create_split(healthy_source, train_healthy, val_healthy, split_ratio=0.7)

    print("\nSplit complete! Check the 'dataset_split' folder.")