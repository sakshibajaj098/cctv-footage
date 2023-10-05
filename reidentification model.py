import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()

        # Define the neural network layers
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 4096),
            nn.Sigmoid(),
        )

        self.embedding_dim = embedding_dim

    def forward_one(self, x):
        # Forward pass through one branch of the Siamese network
        x = self.convolutional(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # Forward pass through both branches of the Siamese network
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Initialize the Siamese Network
siamese_net = SiameseNetwork()
