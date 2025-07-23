import os
from tqdm import tqdm
import shutil # Import shutil for copying

def update_and_create_new_dataset(original_root_dir, new_root_dir):
    """
    Copies the original dataset, updates the annotation files in the new location,
    and sets up the directory structure for the new 3-class dataset.
    """

    # Define the structure for the new dataset
    new_train_images_dir = os.path.join(new_root_dir, 'images', 'train')
    new_val_images_dir = os.path.join(new_root_dir, 'images', 'val')
    new_train_labels_dir = os.path.join(new_root_dir, 'labels', 'train')
    new_val_labels_dir = os.path.join(new_root_dir, 'labels', 'val')

    # Define the structure for the original dataset
    original_train_images_dir = os.path.join(original_root_dir, 'images', 'train')
    original_val_images_dir = os.path.join(original_root_dir, 'images', 'val')
    original_train_labels_dir = os.path.join(original_root_dir, 'labels', 'train')
    original_val_labels_dir = os.path.join(original_root_dir, 'labels', 'val')

    # Create new directories
    os.makedirs(new_train_images_dir, exist_ok=True)
    os.makedirs(new_val_images_dir, exist_ok=True)
    os.makedirs(new_train_labels_dir, exist_ok=True)
    os.makedirs(new_val_labels_dir, exist_ok=True)

    print(f"Copying images from {original_root_dir} to {new_root_dir}...")
    # Copy images (they don't need modification)
    for img_file in tqdm(os.listdir(original_train_images_dir), desc="Copying train images"):
        shutil.copy(os.path.join(original_train_images_dir, img_file), new_train_images_dir)
    for img_file in tqdm(os.listdir(original_val_images_dir), desc="Copying val images"):
        shutil.copy(os.path.join(original_val_images_dir, img_file), new_val_images_dir)

    # Define label directories for processing
    label_mapping_dirs = [
        (original_train_labels_dir, new_train_labels_dir),
        (original_val_labels_dir, new_val_labels_dir)
    ]

    for original_label_dir, new_label_dir in label_mapping_dirs:
        if not os.path.exists(original_label_dir):
            print(f"Warning: Original labels directory not found at {original_label_dir}. Skipping.")
            continue

        print(f"Processing and updating labels from {original_label_dir} to {new_label_dir}")
        for filename in tqdm(os.listdir(original_label_dir), desc=f"Updating {os.path.basename(original_label_dir)} labels"):
            if filename.endswith('.txt'):
                original_filepath = os.path.join(original_label_dir, filename)
                new_filepath = os.path.join(new_label_dir, filename)
                new_lines = []
                
                with open(original_filepath, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue # Skip empty lines

                    original_class_id = int(parts[0])
                    
                    # Map old IDs to new IDs
                    # Only include lines if they belong to our target classes
                    if original_class_id in [0, 5, 6]:  # Enemy (0), Victim (5), Ally (6) -> Person (0)
                        new_class_id = 0
                        new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                    elif original_class_id == 4:  # Door (4) -> Door (1)
                        new_class_id = 1
                        new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                    elif original_class_id == 8:  # Window (8) -> Window (2)
                        new_class_id = 2
                        new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                    # IMPLICIT: If original_class_id is not in [0, 5, 6, 4, 8], the line is NOT added to new_lines,
                    # effectively deleting it from the new annotation file.

                with open(new_filepath, 'w') as f:
                    f.writelines(new_lines)

# Set the original and new dataset root directories
original_dataset_root = '/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_dataset_0723'
new_dataset_root = '/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_3class_dataset_0723'

print(f"Starting dataset modification from '{original_dataset_root}' to '{new_dataset_root}'.")
print(f"This will create a new dataset folder at '{new_dataset_root}' with updated labels.")
update_and_create_new_dataset(original_dataset_root, new_dataset_root)
print("Dataset modification complete. Your new 3-class dataset is ready.")