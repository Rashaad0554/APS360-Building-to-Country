import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Source and destination directories
src_dir = 'filtered_images'  # Folder containing original class folders with images
dst_train = os.path.join('data', 'train')  # Destination for training images
dst_val = os.path.join('data', 'val')      # Destination for validation images
dst_test = os.path.join('data', 'test')    # Destination for test images

# Class names
classes = ['France', 'Greece', 'Italy', 'Japan', 'Mexico']

# Clear out existing images in train and val class folders
for split_dir in [dst_train, dst_val, dst_test]:
    for cls in classes:
        class_dir = os.path.join(split_dir, cls)
        if os.path.exists(class_dir):
            # Remove all files in the class directory
            for f in os.listdir(class_dir):
                file_path = os.path.join(class_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

# For each class, split images and copy to train/val folders
for cls in classes:
    src_cls_dir = os.path.join(src_dir, cls)  # Source directory for this class
    # List all files in the class directory
    images = [f for f in os.listdir(src_cls_dir) if os.path.isfile(os.path.join(src_cls_dir, f))]
    random.shuffle(images)  # Shuffle for random split

    n_total = len(images)
    n_train = int(0.7 * n_total)  # 70% for training
    n_val = int(0.15 * n_total)    # 15% for validation

    train_imgs = images[:n_train]  # First 70% for train
    val_imgs = images[n_train:n_train + n_val]  # Next 15% for val
    test_imgs = images[n_train + n_val:]  # Remaining 15% for test

    # Copy train images to train/class folder
    for img in train_imgs:
        src_path = os.path.join(src_cls_dir, img)
        dst_path = os.path.join(dst_train, cls, img)
        shutil.copy2(src_path, dst_path)

    # Copy val images to val/class folder
    for img in val_imgs:
        src_path = os.path.join(src_cls_dir, img)
        dst_path = os.path.join(dst_val, cls, img)
        shutil.copy2(src_path, dst_path)

    # Copy test images to test/class folder
    for img in test_imgs:
        src_path = os.path.join(src_cls_dir, img)
        dst_path = os.path.join(dst_test, cls, img)
        shutil.copy2(src_path, dst_path)

print("Images have been split and copied to 'data/train', 'data/val', and 'data/test with even class distribution.")