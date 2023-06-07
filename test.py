import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.models import resnet18
from torchvision.models import densenet121
import torchvision.transforms as transforms

batch_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

test_dataset = torchvision.datasets.MNIST(
    root = "./data/", train = False, transform = transforms.ToTensor()
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset, batch_size = batch_size, shuffle = False
)


transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
])

test_dataset1 = torchvision.datasets.MNIST(
    root = "./data/", train = False, transform = transform
)

test_loader1 = torch.utils.data.DataLoader(
    dataset = test_dataset1, batch_size = batch_size, shuffle = False
)



class ConvNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.LazyBatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc=  nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
cnn_model = ConvNet()
cnn_model.load_state_dict(torch.load('cnn.pth'))
cnn_model.to(device)

resnet_model = resnet18(pretrained=False, num_classes=10)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_model.load_state_dict(torch.load('mnist_resnet18.pth'))
resnet_model = resnet_model.to(device)

densenet_model = densenet121(pretrained=False, num_classes=10)
densenet_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
densenet_model.load_state_dict(torch.load('mnist_densenet.pth'))
densenet__model = densenet_model.to(device)

model = [densenet_model, resnet_model, cnn_model]
model_name = ['densenet', 'resnet', 'cnn']

for i in range(3):
    model[i].eval()
    with torch.no_grad():
        correct = 0
        total = 0
        if i == 2:
            for images, lables in test_loader:
                images = images.to(device)
                lables = lables.to(device)
                outputs = model[i](images)
                _, predicted = torch.max(outputs.data, 1)
                total += lables.size(0)
                correct += (predicted == lables).sum().item()
            print("Accuracy: of {} {} % ".format(model_name[i], 100.0 * correct / total))   
        else:
            for images, lables in test_loader1:
                images = images.to(device)
                lables = lables.to(device)
                outputs = model[i](images)
                _, predicted = torch.max(outputs.data, 1)
                total += lables.size(0)
                correct += (predicted == lables).sum().item()
            print("Accuracy of {}: {} % ".format(model_name[i], 100.0 * correct / total)) 