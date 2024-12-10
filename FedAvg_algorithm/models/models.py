from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



class NetworkAttackCNNGRU(nn.Module):
    def __init__(self, input_features, n_classes):
        super(NetworkAttackCNNGRU, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(  1, 32, kernel_size=(3, 1),  stride  = 1 , padding=(1, 0)  )
        self.conv2 = nn.Conv2d( 32, 64, kernel_size=(3, 1),  stride = 1  , padding=(1, 0) )
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.3)
        
        # Calculate flattened feature size
        def conv_output_size(input_size, kernel_size=3, stride=1, padding=1):
            return (input_size + 2*padding - kernel_size) // stride + 1
        
        # Estimate feature size after convolutions
        feature_size = 64 * ((input_features // 2) // 2)
        
        # GRU layer
        self.gru = nn.GRU(feature_size, 128, num_layers=1, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, n_classes)
        
        # Additional dropout
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten and reshape for GRU
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten spatial dimensions
        
        # Reshape for GRU (create a sequence)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # GRU processing
        x, _ = self.gru(x)
        
        # Take the last output of GRU
        x = x[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x