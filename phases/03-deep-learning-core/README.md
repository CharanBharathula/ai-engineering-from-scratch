# Phase 03: Deep Learning Core

This phase transitions from traditional Machine Learning algorithms into Neural Networks, the foundation of modern AI. You will learn PyTorch, the industry standard framework for building deep learning models.

## Roadmap

| Lesson | Description | Status |
|--------|-------------|--------|
| 01. Perceptron & Feedforward Networks | The basic building blocks of neural networks. | ✅ |
| 02. Backpropagation & Autograd | How neural networks learn by calculating gradients. | ⬚ |
| 03. PyTorch Fundamentals | Tensors, Datasets, and DataLoaders in PyTorch. | ⬚ |
| 04. Loss Functions & Optimizers | Cross-entropy, MSE, Adam, and SGD. | ⬚ |
| 05. Regularization & Dropout | Preventing models from memorizing training data. | ⬚ |

## Code Example: A Simple Neural Network in PyTorch

This is the "Hello World" of Deep Learning: building a simple Multi-Layer Perceptron (MLP).

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Neural Network Architecture
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        # First layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function (adds non-linearity)
        self.relu = nn.ReLU()
        # Output layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Initialize Model, Loss Function, and Optimizer
model = SimpleMLP(input_size=10, hidden_size=50, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Dummy Data (Batch of 5 samples, 10 features each)
dummy_inputs = torch.randn(5, 10)
dummy_labels = torch.randint(0, 2, (5,))

# 4. A Single Training Step
# Forward pass
outputs = model(dummy_inputs)
loss = criterion(outputs, dummy_labels)

# Backward pass and optimize
optimizer.zero_grad()  # Clear old gradients
loss.backward()        # Calculate new gradients
optimizer.step()       # Update weights

print(f"Step Loss: {loss.item():.4f}")
```