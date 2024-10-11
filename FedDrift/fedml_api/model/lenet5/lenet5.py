import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet-5 model
class LeNet5_MNIST(torch.nn.Module):
    def __init__(self, feature_dim=784, num_classes=10, input_size=(28, 28)):
        super(LeNet5_MNIST, self).__init__()
        self.num_classes = num_classes
        in_channels = 1 if feature_dim == 784 else 3

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)  # Convolutional layer with 6 feature maps of size 5x5
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 6 feature maps of size 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # Convolutional layer with 16 feature maps of size 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 16 feature maps of size 2x2
        
        # Dinamically calculate the size of the features after convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        self.feature_size = torch.prod(torch.tensor(dummy_output.size()[1:]))

        self.fc1 = nn.Linear(self.feature_size, 120)  # Fully connected layer, output size 120
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer, output size 84
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected layer, output size num_classes

    def forward(self, x):
        # Check the feature dimension (3 channels or 1 channel) to reshape accordingly
        if x.shape[1] == 784:  # Flattened MNIST input (grayscale)
            x = torch.reshape(x, (x.shape[0], 1, 28, 28))  # Reshape to [batch_size, 1, 28, 28]
        elif x.shape[1] == 3 * 28 * 28:  # Flattened 3-channel input (RGB)
            x = torch.reshape(x, (x.shape[0], 3, 28, 28))  # Reshape to [batch_size, 3, 28, 28]

        x = F.relu(self.conv1(x))  # Apply ReLU after conv1
        x = self.pool1(x)  # Apply subsampling pool1
        x = F.relu(self.conv2(x))  # Apply ReLU after conv2
        x = self.pool2(x)  # Apply subsampling pool2
        x_l = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x_l))  # Apply ReLU after fc1
        x = F.relu(self.fc2(x))  # Apply ReLU after fc2
        x = self.fc3(x)  # Output layer
        
        return x
    

# LeNet-5 model
class LeNet5_CIFAR(torch.nn.Module):
    def __init__(self, feature_dim=1024, num_classes=10, input_size=(32, 32)):
        super(LeNet5_CIFAR, self).__init__()
        self.num_classes = num_classes
        in_channels = 1 if feature_dim == 1024 else 3

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)  # Convolutional layer with 6 feature maps of size 5x5
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 6 feature maps of size 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # Convolutional layer with 16 feature maps of size 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 16 feature maps of size 2x2
        
        # Dinamically calculate the size of the features after convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        self.feature_size = torch.prod(torch.tensor(dummy_output.size()[1:]))

        self.fc1 = nn.Linear(self.feature_size, 120)  # Fully connected layer, output size 120
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer, output size 84
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected layer, output size num_classes

    def forward(self, x):
        # Check the feature dimension (3 channels or 1 channel) to reshape accordingly
        if x.shape[1] == 1024:  # Flattened MNIST input (grayscale)
            x = torch.reshape(x, (x.shape[0], 1, 32, 32))  # Reshape to [batch_size, 1, 28, 28]
        elif x.shape[1] == 3 * 32 * 32:  # Flattened 3-channel input (RGB)
            x = torch.reshape(x, (x.shape[0], 3, 32, 32))  # Reshape to [batch_size, 3, 28, 28]

        x = F.relu(self.conv1(x))  # Apply ReLU after conv1
        x = self.pool1(x)  # Apply subsampling pool1
        x = F.relu(self.conv2(x))  # Apply ReLU after conv2
        x = self.pool2(x)  # Apply subsampling pool2
        x_l = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x_l))  # Apply ReLU after fc1
        x = F.relu(self.fc2(x))  # Apply ReLU after fc2
        x = self.fc3(x)  # Output layer
        
        return x