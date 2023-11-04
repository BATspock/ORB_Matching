import os
import glob
import shutil

def get_images(directory, num_images = 4, num_subfolders = 5):
    # list images in folders
    subfolders = sorted([f.path for f in os.scandir(directory) if f.is_dir()])
    required_folders = subfolders[:num_subfolders]
    # display images in the images subfolder
    for folder in required_folders:
        # get folder name
        folder_name = folder.split("\\")[-1]
        print(folder_name)
        images_folder = os.path.join(folder, 'images')
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
        # copy images to new folder 'images' in the root directory
        images = image_files[:num_images]
        # number the images 
        i = 1
        for img in images:
            shutil.copy2(img, os.path.join('images', f'{folder_name}_{i}.jpg'))
            i+=1
           
# create a folder named images
if not os.path.exists('images'):
    os.mkdir('images')
get_images('image-matching-challenge-2022/train', 4, 5)

