# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt

# PyTorch dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast

import sys

def device_check():
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    return train_on_gpu

def data_loading(num_workers = 8, batch_size = 256):
    
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=train_transform)
    test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=test_transform)

    return train_data, test_data

def training_indices(num_train, valid_size = 0.2):
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx

def samplers(train_idx, valid_idx):
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler

def prep_loaders(train_data, train_sampler, valid_sampler, test_data, batch_size = 256, num_workers = 8):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)
    return train_loader, valid_loader, test_loader

def specify_classes():
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return classes

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
    def __init__(self, block, layers, num_classes = 10):
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
    
def training_and_validation(model, train_loader, valid_loader, train_on_gpu, optimizer, criterion, scaler):
    # number of epochs to train the model
    n_epochs = 10

    valid_loss_min = np.inf # track change in validation loss

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
    
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):                          # forward pass in FP16
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()            # scaled backward pass
            scaler.step(optimizer)                   # unscale + optimizer step
            scaler.update()                          # adjust scale factor
            train_loss += loss.item() * data.size(0)
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            with autocast(dtype=torch.bfloat16):
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
    
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
    
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss

    return None


def main():
    train_on_gpu = device_check()
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    
    model = ResNet(ResidualBlock, [3, 4, 6, 3])
    print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    
    if train_on_gpu:
        model = model.cuda()
    print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")

    criterion = nn.CrossEntropyLoss() #define categorical cross-entropy loss function
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9) #define the optimizer with learning rate
    print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")

    train_data, test_data = data_loading()
    print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

    train_idx, valid_idx = training_indices(num_train = len(train_data))

    train_sampler, valid_sampler = samplers(train_idx, valid_idx)

    train_loader, valid_loader, test_loader = prep_loaders(train_data=train_data, train_sampler=train_sampler, valid_sampler=valid_sampler, test_data=test_data)

    classes = specify_classes()
    
    scaler = GradScaler()

    training_and_validation(model=model, train_loader=train_loader, valid_loader=valid_loader, train_on_gpu=train_on_gpu, optimizer=optimizer, criterion=criterion, scaler=scaler)

    return 0
    
if __name__ == "__main__":
    sys.exit(main())