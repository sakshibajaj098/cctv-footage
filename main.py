def evaluate_model(test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            img1, img2, labels = data
            output1, output2 = siamese_net(img1, img2)
            distances = nn.functional.pairwise_distance(output1, output2)
            predicted = (distances < 0.5).float()  # Assuming a threshold of 0.5

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Assume test_loader is the DataLoader for the test set
evaluate_model(test_loader)
