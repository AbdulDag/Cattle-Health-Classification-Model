import os
import uuid

def reorder_and_rename(folder_path, suffix):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Could not find the folder '{folder_path}'")
        return

    files = os.listdir(folder_path)
    temp_files = []
    
    print(f"\nProcessing '{folder_path}'...")
    
    #Rename everything to a temporary random name to avoid name collision
    for filename in files:
        old_path = os.path.join(folder_path, filename)
        
        # Skip subfolders
        if os.path.isdir(old_path):
            continue
            
        extension = os.path.splitext(filename)[1].lower()
        # Create a completely unique temporary name
        temp_name = f"temp_{uuid.uuid4().hex}{extension}"
        temp_path = os.path.join(folder_path, temp_name)
        
        os.rename(old_path, temp_path)
        temp_files.append(temp_name)
        
    #Rename all temporary files into sequential order
    counter = 1
    for temp_filename in temp_files:
        temp_path = os.path.join(folder_path, temp_filename)
        extension = os.path.splitext(temp_filename)[1].lower()
        
        new_name = f"{counter}_{suffix}{extension}"
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(temp_path, new_path)
        counter += 1

    print(f"Successfully reordered {counter - 1} images!")

if __name__ == "__main__":
    base_dir = "cow_dataset"
    sick_dir = os.path.join(base_dir, "1_Sick")
    healthy_dir = os.path.join(base_dir, "0_Healthy")

    print("Starting reordering process...")
    
    # Process both folders
    reorder_and_rename(sick_dir, "sick")
    reorder_and_rename(healthy_dir, "healthy")
    
    print("\nThe dataset numbering is perfectly continuous.")