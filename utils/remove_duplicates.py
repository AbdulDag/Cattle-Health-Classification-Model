import os
from PIL import Image
import imagehash

def clean_duplicates(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Dictionary to store the visual "fingerprints" of the images
    seen_hashes = {}
    deleted_count = 0
    
    print(f"\n--- Scanning for duplicates in {folder_path} ---")
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Skip subfolders
        if os.path.isdir(filepath):
            continue
            
        try:
            # Open the image and calculate its Perceptual Hash (phash)
            with Image.open(filepath) as img:
                # phash looks at the visual features, not just the raw file data
                img_hash = str(imagehash.phash(img))
                
            # Check if we have seen this exact visual pattern before
            if img_hash in seen_hashes:
                print(f"Deleting duplicate: {filename} (Looks identical to {seen_hashes[img_hash]})")
                os.remove(filepath)
                deleted_count += 1
            else:
                # Remember this fingerprint for the next images
                seen_hashes[img_hash] = filename
                
        except Exception as e:
            print(f"Could not process {filename}: {e}")

    print(f"Finished! Deleted {deleted_count} duplicates from '{folder_path}'.")

# --- EXECUTION ---
if __name__ == "__main__":
    base_dir = "cow_dataset"
    sick_dir = os.path.join(base_dir, "1_Sick")
    healthy_dir = os.path.join(base_dir, "0_Healthy")

    # Run the scanner on both folders
    clean_duplicates(sick_dir)
    clean_duplicates(healthy_dir)