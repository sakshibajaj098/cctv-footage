def demonstrate_reidentification(input_image, dataset_images, dataset_distances, top_k=5):
    # Compute distances/similarities between input_image and dataset images
    input_embedding = siamese_net.forward_one(input_image)
    distances = np.linalg.norm(dataset_distances - input_embedding, axis=1)

    # Get indices of top-k closest matches
    top_k_indices = np.argsort(distances)[:top_k]

    # Display the input image and the top-k closest matches
    plt.figure(figsize=(12, 4))
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(input_image.squeeze().permute(1, 2, 0))
    plt.title('Input Image')

    for i, idx in enumerate(top_k_indices):
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(dataset_images[idx].squeeze().permute(1, 2, 0))
        plt.title(f'Distance: {distances[idx]:.4f}')

    plt.show()

# Assume dataset_images contains the dataset images and dataset_distances contains their embeddings
# Also, assume input_image contains the image to demonstrate re-identification
demonstrate_reidentification(input_image, dataset_images, dataset_distances)
