import torch
import torch.nn as nn
import torch.nn.functional as F

# Resnet-9 layer
def residual_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ResNet-9 model
class ResNet9_MNIST(nn.Module):
    def __init__(self, feature_dim, num_classes, input_size=(28, 28)):
        super().__init__()
        in_channels = 1 if feature_dim == 784 else 3
        self.num_classes = num_classes
        self.prep = residual_block(in_channels, 64)
        self.layer1_head = residual_block(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(residual_block(128, 128), residual_block(128, 128))
        self.layer2 = residual_block(128, 256, pool=True)
        self.layer3_head = residual_block(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(residual_block(512, 512), residual_block(512, 512))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed to adaptive average pooling:         self.MaxPool2d = nn.Sequential(nn.MaxPool2d(4))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the features after the convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool(self.layer3_head(self.layer2(self.layer1_head(self.prep(dummy_input)))))
        self.feature_size = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)

        # Output layer
        self.linear = nn.Linear(self.feature_size, num_classes)

    def forward(self, x):
        # Check the feature dimension (3 channels or 1 channel) to reshape accordingly
        if x.shape[1] == 784:  # Flattened MNIST input (grayscale)
            x = torch.reshape(x, (x.shape[0], 1, 28, 28))  # Reshape to [batch_size, 1, 28, 28]
        elif x.shape[1] == 3 * 28 * 28:  # Flattened 3-channel input (RGB)
            x = torch.reshape(x, (x.shape[0], 3, 28, 28))  # Reshape to [batch_size, 3, 28, 28]

        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.pool(x)  # Changed to adaptive average pooling
        x_l = x.view(x.size(0), -1)
        x = self.linear(x_l)
        return x
    

# ResNet-9 model
class ResNet9_CIFAR(nn.Module):
    def __init__(self, feature_dim, num_classes, input_size=(32, 32)):
        super().__init__()
        in_channels = 1 if feature_dim == 1024 else 3
        self.num_classes = num_classes
        self.prep = residual_block(in_channels, 64)
        self.layer1_head = residual_block(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(residual_block(128, 128), residual_block(128, 128))
        self.layer2 = residual_block(128, 256, pool=True)
        self.layer3_head = residual_block(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(residual_block(512, 512), residual_block(512, 512))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed to adaptive average pooling:         self.MaxPool2d = nn.Sequential(nn.MaxPool2d(4))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the features after the convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool(self.layer3_head(self.layer2(self.layer1_head(self.prep(dummy_input)))))
        self.feature_size = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)

        # Output layer
        self.linear = nn.Linear(self.feature_size, num_classes)

    def forward(self, x):
        # Check the feature dimension (3 channels or 1 channel) to reshape accordingly
        if x.shape[1] == 784:  # Flattened MNIST input (grayscale)
            x = torch.reshape(x, (x.shape[0], 1, 32, 32))  # Reshape to [batch_size, 1, 28, 28]
        elif x.shape[1] == 3 * 32 * 32:  # Flattened 3-channel input (RGB)
            x = torch.reshape(x, (x.shape[0], 3, 32, 32))  # Reshape to [batch_size, 3, 28, 28]

        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.pool(x)  # Changed to adaptive average pooling
        x_l = x.view(x.size(0), -1)
        x = self.linear(x_l)
        return x