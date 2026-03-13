import os
import shutil
from bing_image_downloader import downloader
from PIL import Image

#same as get_cows.py, just diff queries
base_dir = "cow_dataset_batch_2"
sick_dir = os.path.join(base_dir, "1_Sick")
healthy_dir = os.path.join(base_dir, "0_Healthy")


sick_queries = [
    "bovine cachexia",              
    "cow spine visible",            
    "cattle severe weight loss",
    "cow showing ribs and hip bones",
    "cow standing with arched back", 
    "lame cow walking",             
    "cow head pressing",            
    "sick cow isolated in pen",     
    "cow sunken eyes",              
    "calf with droopy ears",
    "cow rough hair coat"          
]

healthy_queries = [
    # Different breeds and angles to make the model robust
    "healthy jersey cow standing", 
    "brown swiss cattle grazing", 
    "healthy angus bull", 
    "simmental cow pasture",
    
    # to teach model move = healthy
    "cow walking firmly",
    "happy cows running",
    "herd of cattle eating",
    "cow looking alert"
]

#SAFETY FUNCTION
def verify_images(folder_path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    print(f"--- Verifying safety of images in {folder_path} ---")
    
    if not os.path.exists(folder_path):
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in valid_exts:
            try:
                os.remove(file_path)
            except: pass
            continue
            
        try:
            with Image.open(file_path) as img:
                img.verify()
        except:
            try:
                print(f"Deleting corrupt file: {filename}")
                os.remove(file_path)
            except: pass

#DOWNLOADER FUNCTION
def download_and_clean(queries, target_folder):
    for query in queries:
        print(f"Downloading Batch 2: {query}...")
        try:
            downloader.download(
                query, 
                limit=30,  # 30 images per term x 12 terms = ~360 new images
                output_dir=base_dir, 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=10, 
                verbose=False
            )
            
            # Move files
            source_folder = os.path.join(base_dir, query)
            if os.path.exists(source_folder):
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    
                for file_name in os.listdir(source_folder):
                    src = os.path.join(source_folder, file_name)
                    dst = os.path.join(target_folder, file_name)
                    
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(file_name)
                        dst = os.path.join(target_folder, f"{base}_{query}{ext}")
                        
                    shutil.move(src, dst)
                os.rmdir(source_folder)
        except Exception as e:
            print(f"Skipped {query}: {e}")
    
    verify_images(target_folder)

if __name__ == "__main__":
    print("Starting BATCH 2 download for SICK cows...")
    download_and_clean(sick_queries, sick_dir)

    print("\nStarting BATCH 2 download for HEALTHY cows...")
    download_and_clean(healthy_queries, healthy_dir)

    print("\nDone! Check the 'cow_dataset_batch_2' folder.")