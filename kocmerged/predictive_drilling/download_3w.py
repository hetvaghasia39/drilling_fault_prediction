import kagglehub
import os
import shutil

# Download 3W dataset
print("Downloading 3W dataset...")
path = kagglehub.dataset_download("afrniomelo/3w-dataset")
print(f"Dataset downloaded to: {path}")

# Move to project directory
target_dir = os.path.abspath("3w/data/raw/3w_official")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f"Copying files to {target_dir}...")
# Copy logic depends on folder structure, usually it's a directory
if os.path.isdir(path):
    # copy tree content
    import distutils.dir_util
    distutils.dir_util.copy_tree(path, target_dir)
else:
    shutil.copy2(path, target_dir)

print("Download and copy complete.")
