import numpy as np

def get_observation(x, y):
    # Placeholder for the simulated observation function
    return x + y

def localize_robot(floorplan, observations):
    """
    Localizes the robot in a 2D floorplan using a particle filter.

    :param floorplan: A 2D array representing the floorplan.
    :param observations: The latest image-based observations.
    :return: The estimated coordinates (x, y) of the robot.
    """

    # Constants
    num_particles = 1000
    noise = 0.1

    # Initialize particles randomly
    particles = np.random.rand(num_particles, 2) * np.array(floorplan.shape)
    
    # Update particles based on observations
    weights = np.ones(num_particles)
    for obs in observations:
        for i in range(num_particles):
            x, y = particles[i]
            x = int(x)
            y = int(y)
            if 0 <= x < floorplan.shape[0] and 0 <= y < floorplan.shape[1] and floorplan[x, y] == 0:
                # Update weight based on observation likelihood
                expected_obs = get_observation(x, y)
                weights[i] *= np.exp(-0.5 * ((obs - expected_obs) / noise) ** 2)
            else:
                # Particle is in obstacle, set weight to zero
                weights[i] = 0

    # Normalize weights
    weights /= np.sum(weights)
    
    # Resample particles
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices]
    
    # Estimate position as the mean of the particles
    estimate = np.mean(particles, axis=0)
    
    return estimate

# Create a dummy 5x5 floorplan
# 0 represents free space, 1 represents obstacles
floorplan = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Create some dummy observations
# Let's say the robot observes the values 6, 4, and 2 at its current position in sequence.
observations = [6, 4, 2]

# Localize the robot
estimate = localize_robot(floorplan, observations)
print(f'Estimated Position: {estimate}')
