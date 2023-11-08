import numpy as np
import cv2

class RobotLocalizer:
    def __init__(self, floorplan):
        self.floorplan = floorplan
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.precomputed_descriptors = self._compute_descriptors()

    def _compute_descriptors(self):
        """Precomputes ORB descriptors for each free space in the floorplan."""
        descriptors = {}
        for i in range(self.floorplan.shape[0]):
            for j in range(self.floorplan.shape[1]):
                if self.floorplan[i, j] == 0:
                    # The dummy image in real life scenario will be images of 
                    # the floor plan at coordinate (i,j)
                    dummy_image = np.random.randint(255, size=(100, 100), dtype=np.uint8)
                    _, descriptor = self.orb.detectAndCompute(dummy_image, None)
                    descriptors[(i, j)] = descriptor
        return descriptors

    def localize_robot(self, observation):
        """
        Localizes the robot in a 2D floorplan using image-based ORB descriptors and RANSAC for robust matching.
        
        :param observation: The latest image captured by the robot.
        :return: The estimated (x, y) coordinates of the robot, or None if no match found.
        """
        # Find keypoints and descriptors in the observation image
        kp_observation, obs_descriptors = self.orb.detectAndCompute(observation, None)

        best_match_location = None
        highest_num_inliers = -1

        # Iterate over the precomputed descriptors to find the best match using RANSAC
        for location, precomp_descriptor in self.precomputed_descriptors.items():
            # Match descriptors using the Brute Force Matcher
            matches = self.bf.knnMatch(obs_descriptors, precomp_descriptor, k=2)

            # Apply Lowe's ratio test to find good matches
            good_matches = [m for m, n in matches if len(n) == 2 and m.distance < 0.75 * n.distance]

            if len(good_matches) >= 4:  # Minimum number of matches to compute homography
                # Extract location of good matches
                points_observation = np.float32([kp_observation[m.queryIdx].pt for m in good_matches])
                points_precomputed = np.float32([(location[1], location[0]) for m in good_matches])

                # Compute homography using RANSAC
                _, mask = cv2.findHomography(points_observation, points_precomputed, cv2.RANSAC)

                # Count the number of inliers
                num_inliers = np.sum(mask)

                # Update the best match if the current one has more inliers
                if num_inliers > highest_num_inliers:
                    highest_num_inliers = num_inliers
                    best_match_location = location

        return best_match_location


# Example usage:
# Create a dummy 5x5 floorplan with free space (0) and obstacles (1)
floorplan = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Initialize the localizer with the floorplan
localizer = RobotLocalizer(floorplan)

# Assuming 'observation_image' is the current image captured by the robot's camera
# observation_image = cv2.imread('path_to_the_image.jpg', 0)  # Read as grayscale

# In this example, since we don't have an actual image, let's create a dummy image and use its descriptor
dummy_location = (2, 2)  # A known location
dummy_image = np.random.randint(255, size=(100, 100), dtype=np.uint8)  # Placeholder for an actual image
_, dummy_descriptor = localizer.orb.detectAndCompute(dummy_image, None)
localizer.precomputed_descriptors[dummy_location] = dummy_descriptor

# Now, localize the robot with the dummy observation
estimated_position = localizer.localize_robot(dummy_image)
print(f'Estimated Position of the Robot: {estimated_position}')
