import torch
import torch.nn as nn

class CoughDetectionModel(nn.Module):
    def __init__(self):
        super(CoughDetectionModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Use the correct dummy input size used during training
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)  # <- Corrected shape
            x = self.maxpool1(self.relu1(self.conv1(dummy_input)))
            x = self.maxpool2(self.relu2(self.conv2(x)))
            flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
