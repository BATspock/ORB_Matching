# 1. A ImageMatcher class is created to encapsulate the functionality.
# 2. The __init__ method initializes the class with the specified images 
#     folder path, an ORB detector, a BFMatcher instance, and an empty 
#     dictionary for storing image descriptors.
# 3. The load_images_and_compute_descriptors method loads the images from 
#     the specified folder, converts them to grayscale, and computes ORB
#     descriptors for each image.
# 4. The compare_images method compares every image to every other image 
#     and stores the number of good matches in a dictionary.
# 5. The find_most_similar_images method sorts the pairs of images based on 
#     the number of good matches and returns the top N pairs.
# 6. In the usage section, an ImageMatcher instance is created, the 
#     load_images_and_compute_descriptors and find_most_similar_images methods
#     are called, and the pairs of images with the most matches are printed.

import cv2
import os
import glob
from collections import defaultdict
from operator import itemgetter
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

class ImageMatcher:
    def __init__(self, images_folder_path):
        """
        Initializes the ImageMatcher with the specified images folder path.

        :param images_folder_path: Path to the folder containing images.
        """
        self.images_folder_path = images_folder_path
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        self.image_descriptors = {}

    def load_images_and_compute_descriptors(self):
        """
        Loads images from the specified folder and computes ORB descriptors for each image.
        """
        for image_path in glob.glob(os.path.join(self.images_folder_path, '*.jpg')):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            self.image_descriptors[image_path] = des

    def compare_images(self):
        """
        Compares every image to every other image and stores the number of good matches.

        :return: A dictionary containing the number of good matches for each pair of images.
        """
        matches_dict = defaultdict(list)
        for image_path_1 in self.image_descriptors.keys():
            for image_path_2 in self.image_descriptors.keys():
                if image_path_1 != image_path_2:
                    matches = self.bf.knnMatch(self.image_descriptors[image_path_1], self.image_descriptors[image_path_2], k=2)
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append([m])
                    matches_dict[(image_path_1, image_path_2)] = len(good_matches)
        return matches_dict

    def find_most_similar_images(self, top_n=10):
        """
        Finds the most similar images based on the number of good matches.

        :param top_n: Number of top similar image pairs to return.
        :return: A sorted list of tuples containing the pairs of images and number of good matches.
        """
        matches_dict = self.compare_images()
        sorted_matches = sorted(matches_dict.items(), key=itemgetter(1), reverse=True)
        return sorted_matches[:top_n]
    

# create a folder named images
if not os.path.exists('images'):
    os.mkdir('images')
get_images('image-matching-challenge-2022/train', 4, 5)

# Usage pf ImageMatcher class
image_matcher = ImageMatcher('images')
image_matcher.load_images_and_compute_descriptors()
most_similar_images = image_matcher.find_most_similar_images()

# Print the pairs of images with the most matches
for pair, num_matches in most_similar_images:
    print(pair, num_matches)
