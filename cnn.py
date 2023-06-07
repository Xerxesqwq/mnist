import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

num_epochs = 10
batch_size = 256
lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

train_dataset = torchvision.datasets.MNIST(
    root = "./data/", train = True, transform = transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    root = "./data/", train = False, transform = transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset, batch_size = batch_size,shuffle = True 
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset, batch_size = batch_size, shuffle = False
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
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch[{}/{}]. Step[{}/{}], Loss:{:.4f}".format(
        epoch + 1, num_epochs, i + 1, total_step, loss.item()
    ))

torch.save(model.state_dict(), 'cnn.pth')
print('Finished Training and Saving the model')