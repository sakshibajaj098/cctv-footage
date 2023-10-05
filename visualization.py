import matplotlib.pyplot as plt
import numpy as np

# Function to display image pairs with their distance
def display_image_pairs(image1, image2, distance):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title('Image 2')

    plt.suptitle(f'Distance: {distance:.4f}')
    plt.show()

# Generate a sample image pair and distance
sample_image1 = np.random.rand(100, 100, 3)  # Replace with actual image data
sample_image2 = np.random.rand(100, 100, 3)  # Replace with actual image data
sample_distance = np.random.rand()  # Replace with actual distance computed by the model

# Display the sample image pair
display_image_pairs(sample_image1, sample_image2, sample_distance)
