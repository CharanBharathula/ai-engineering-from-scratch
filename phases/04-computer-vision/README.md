# Phase 04: Computer Vision

Computer Vision (CV) gives AI the ability to process and understand visual data. In this phase, we move from basic image processing to Convolutional Neural Networks (CNNs).

## Roadmap

| Lesson | Description | Status |
|--------|-------------|--------|
| 01. Image Processing Basics | OpenCV, filters, and edge detection. | ✅ |
| 02. Convolutional Neural Networks (CNN) | Convolutions, pooling, and feature maps. | ⬚ |
| 03. Modern Architectures | ResNet, VGG, and MobileNet. | ⬚ |
| 04. Object Detection | YOLO, Faster R-CNN, and bounding boxes. | ⬚ |
| 05. Image Segmentation | Mask R-CNN and U-Net. | ⬚ |

## Code Example: Building a CNN for Image Classification

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Classifier
        # Assuming input image is 32x32. After two MaxPools (divide by 2 twice), size is 8x8.
        # 32 channels * 8 * 8 = 2048 flattened features
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 classes

    def forward(self, x):
        # Extract features
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten for Dense layer
        x = x.view(x.size(0), -1) 
        
        # Classify
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Test with a dummy image tensor (Batch Size=1, Channels=3, Height=32, Width=32)
model = SimpleCNN()
dummy_image = torch.randn(1, 3, 32, 32)
output = model(dummy_image)

print(f"Output shape: {output.shape}") # Expected: [1, 10]
```