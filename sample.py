import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import AdamW
from torchvision import datasets, transforms

# Define hyperparameters
input_size = 3 * 32 * 32  # CIFAR10 images are 3x32x32
hidden_size = 128
output_size = 10  # CIFAR10 has 10 classes
learning_rate = 0.001


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Using nn.Sequential
model = nn.Sequential(
    nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)
)

model = SimpleNN(input_size, hidden_size, output_size)
optimizer = AdamW(model.parameters(), lr=learning_rate)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model.train()
for inputs, labels in train_loader:
    # Flatten the inputs for the fully connected network
    inputs = inputs.view(inputs.size(0), -1)
    predictions = model(inputs)
    loss = F.cross_entropy(predictions, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
