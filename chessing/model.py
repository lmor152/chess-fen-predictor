import torch.nn as nn
import torch.nn.functional as F

from chessing import LABELS


class ChessPieceClassifier(nn.Module):
    def __init__(self):
        super(ChessPieceClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * 4 * 4, len(LABELS))

    def forward(self, x):
        # x should have shape [batch_size, num_squares, 1, 50, 50] (16, 64, 1, 50, 50)
        batch_size, num_squares, channels, height, width = x.shape

        # Flatten the squares
        x = x.view(
            batch_size * num_squares, channels, height, width
        )  # Shape: [batch_size*num_squares, 1, 50, 50]

        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(
            batch_size * num_squares, -1
        )  # Shape: [batch_size*num_squares, features]

        x = self.dropout(x)  # Apply Dropout
        x = self.fc1(x)  # Fully connected layer
        x = x.view(
            batch_size, num_squares, -1
        )  # Reshape back to [batch_size, num_squares, len(LABELS)]

        return x
