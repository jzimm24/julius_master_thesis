# test_model.py
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import sys

# Copy the model architecture (required to load the weights)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.maxpool = nn.Identity()
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Device
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if train_on_gpu else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = ResNet(ResidualBlock, [3, 4, 6, 3])
    model.load_state_dict(torch.load('model_cifar.pt', map_location=device))
    model.to(device)
    model.eval()
    print('Model loaded successfully.')

    # Test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, num_workers=8)

    # Test loop
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            # Per-class accuracy
            _, pred = torch.max(output, 1)
            correct = pred.eq(target)
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Overall accuracy
    test_loss /= len(test_loader.dataset)
    overall_acc = 100. * sum(class_correct) / sum(class_total)
    print(f'\nTest Loss: {test_loss:.6f}')
    print(f'Overall Accuracy: {overall_acc:.2f}%\n')

    # Per-class accuracy
    for i in range(10):
        acc = 100. * class_correct[i] / class_total[i]
        print(f'Accuracy of {classes[i]:>12s}: {acc:.2f}%  ({class_correct[i]}/{class_total[i]})')

    return 0

if __name__ == '__main__':
    sys.exit(main())