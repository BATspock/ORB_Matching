import numpy as np
import cv2

class RobotLocalizer:
    def __init__(self, floorplan):
        self.floorplan = floorplan
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.precomputed_descriptors = self._compute_descriptors()
        # Buffer to store the last three positions
        self.positions_buffer = []  

    def _compute_descriptors(self):
        """Precomputes ORB descriptors for each free space in the floorplan."""
        descriptors = {}
        for i in range(self.floorplan.shape[0]):
            for j in range(self.floorplan.shape[1]):
                if self.floorplan[i, j] == 0:
                    # In a real scenario, you would extract the descriptor from an actual image
                    # Here, we're just creating dummy images and descriptors for demonstration
                    dummy_image = np.zeros((100, 100), dtype=np.uint8)
                    _, descriptor = self.orb.detectAndCompute(dummy_image, None)
                    descriptors[(i, j)] = descriptor
        return descriptors

  def update_position_buffer(self, new_position):
        """
        Update the buffer with the new position and keep only the last three positions.
        """
        self.positions_buffer.append(new_position)
        if len(self.positions_buffer) > 3:
            self.positions_buffer.pop(0)

  def get_average_position(self):
          """
          Calculate the average of the last three positions in the buffer.
          """
          if not self.positions_buffer:
              return None
          # Calculate average for x and y separately
          x_avg = sum(position[0] for position in self.positions_buffer) / len(self.positions_buffer)
          y_avg = sum(position[1] for position in self.positions_buffer) / len(self.positions_buffer)
          return x_avg, y_avg

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
localizer = RobotLocalizer(floorplan)

 # Simulate a stream of images
for i in range(10):  # Replace with actual image stream in real application
    # Simulate capturing a new image
    observation_image = np.random.randint(255, size=(100, 100), dtype=np.uint8)  # Replace with actual image capture

    # Process the image to get the current position
    current_position = localizer.localize_robot(observation_image)
    if current_position:
        localizer.update_position_buffer(current_position)

        # Calculate the average position from the last three positions
        avg_position = localizer.get_average_position()
        if avg_position:
            print(f'Current average position: {avg_position}')
        else:
            print('Not enough data to calculate average position')
    else:
        print('Localization failed for the current image')
