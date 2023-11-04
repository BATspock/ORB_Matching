import matplotlib.pyplot as plt
import numpy as np
import time
import os
import glob
from image_matching import ImageMatcher
from image_matching import get_images

def analyze_performance(max_subfolders):
    subfolder_counts = range(5, max_subfolders + 1)
    execution_times = []
    average_confidences = []

    for subfolder_count in subfolder_counts:
        # Prepare images
        if not os.path.exists('images'):
            os.mkdir('images')
        get_images('image-matching-challenge-2022/train', 4, subfolder_count)

        # Measure execution time
        start_time = time.time()
        image_matcher = ImageMatcher('images')
        image_matcher.load_images_and_compute_descriptors()
        most_similar_images = image_matcher.find_most_similar_images()
        execution_times.append(time.time() - start_time)

        # Compute average confidence of matches
        total_confidence = 0
        total_matches = 0
        for _, num_matches in most_similar_images:
            total_confidence += num_matches
            total_matches += 1
        average_confidences.append(total_confidence / total_matches if total_matches > 0 else 0)

        # Clear images folder for next iteration
        for file in glob.glob('images/*.jpg'):
            os.remove(file)

    # Plot results
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of Subfolders')
    ax1.set_ylabel('Execution Time (s)', color=color)
    ax1.plot(subfolder_counts, execution_times, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average Confidence', color=color)
    ax2.plot(subfolder_counts, average_confidences, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

# Call the function to analyze performance
analyze_performance(15)
