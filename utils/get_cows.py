import os
import shutil
from bing_image_downloader import downloader
from PIL import Image  # This is the safety inspector


sick_queries = ["emaciated cow", "sick cow lying down", "cow with ribs showing", "ill cattle"]
healthy_queries = ["healthy dairy cow standing", "fat cow grazing", "prize winning cattle", "healthy bull"]

# Where to save (This will create the folder inside your Documents folder)
base_dir = "cow_dataset"
sick_dir = os.path.join(base_dir, "1_Sick")
healthy_dir = os.path.join(base_dir, "0_Healthy")

#Safety Function
def verify_images(folder_path):
    """
    Scans a folder and deletes any file that is:
    1. Not a .jpg, .jpeg, or .png
    2. Corrupt or not a real image (virus protection)
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    print(f"--- Verifying safety of images in {folder_path} ---")
    
    # Check if folder exists first
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check 1: File Extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_exts:
            try:
                print(f"Deleting (bad extension): {filename}")
                os.remove(file_path)
            except:
                pass
            continue
            
        # Check 2: Deep Inspection (The Virus Checker)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Checks if the file is broken or fake
        except (IOError, SyntaxError) as e:
            try:
                print(f"Deleting (corrupt/fake file): {filename}")
                os.remove(file_path)
            except:
                pass

#Downloader Function
def download_and_clean(queries, target_folder):
    # 1. Download to temporary folders
    for query in queries:
        print(f"Downloading: {query}...")
        try:
            downloader.download(
                query, 
                limit=30, 
                output_dir=base_dir, 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=10, 
                verbose=False
            )
            
            # 2. Move to our main folder
            # The downloader creates a folder named after the query inside base_dir
            source_folder = os.path.join(base_dir, query)
            
            if os.path.exists(source_folder):
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    
                for file_name in os.listdir(source_folder):
                    src = os.path.join(source_folder, file_name)
                    dst = os.path.join(target_folder, file_name)
                    
                    # Avoid overwriting names
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(file_name)
                        dst = os.path.join(target_folder, f"{base}_{query}{ext}")
                        
                    shutil.move(src, dst)
                    
                # 3. Remove the temp folder
                os.rmdir(source_folder)
        except Exception as e:
            print(f"Skipping {query} due to error: {e}")
    
    # 4. Run the Safety Check
    verify_images(target_folder)

if __name__ == "__main__":
    print("Starting SAFE download for SICK cows...")
    download_and_clean(sick_queries, sick_dir)

    print("\nStarting SAFE download for HEALTHY cows...")
    download_and_clean(healthy_queries, healthy_dir)

    print("\nDone! Check the 'cow_dataset' folder.")