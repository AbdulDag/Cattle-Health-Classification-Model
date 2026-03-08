import os

def rename_images(folder_path, suffix):
    # Check if the folder actually exists
    if not os.path.exists(folder_path):
        print(f"Error: Could not find the folder '{folder_path}'")
        return

    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    counter = 1
    for filename in files:
        old_path = os.path.join(folder_path, filename)
        
        # Skip if it's a subfolder
        if os.path.isdir(old_path):
            continue
            
        # Get the file extension
        extension = os.path.splitext(filename)[1].lower()
        
        # Create the new filename
        new_name = f"{counter}_{suffix}{extension}"
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        try:
            os.rename(old_path, new_path)
            counter += 1
        except FileExistsError:
            print(f"Warning: {new_name} already exists. Skipping.")
        except Exception as e:
            print(f"Could not rename {filename}: {e}")

    print(f"Successfully renamed {counter - 1} images in '{folder_path}'")

if __name__ == "__main__":
    base_dir = "cow_dataset"
    sick_dir = os.path.join(base_dir, "1_Sick")
    healthy_dir = os.path.join(base_dir, "0_Healthy")

    print("Starting renaming process...")
    
    # Rename the Sick folder images
    rename_images(sick_dir, "sick")
    
    # Rename the Healthy folder images
    rename_images(healthy_dir, "healthy")
    
    print("\nThe folders are now clean and organized.")