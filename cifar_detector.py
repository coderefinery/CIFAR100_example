import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define a Convolutional Neural Network
class detector(nn.Module):
    def __init__(self):
        super(detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc = nn.Linear(128, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


# Load and transform the CIFAR10 dataset
train_set = torchvision.datasets.CIFAR100(
    root='./data', train=True,
    download=True, transform=transforms.ToTensor()
)
test_set = torchvision.datasets.CIFAR100(
    root='./data', train=False,
    download=True, transform=transforms.ToTensor()
)


# Initialize the network and optimizer
net = detector().to(device)
optimizer = torch.optim.Adam(net.parameters())

# Train the network
for epoch in range(10):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, batch {i}, loss: {loss.item()}')

