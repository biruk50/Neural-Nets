import os
import random
import numpy as np
import torch
from torch import device, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class LightweightCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2)
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.classifier(x)

def create_model_rgb(num_classes):
    return LightweightCNN(3, num_classes)

def create_model_gray(num_classes):
    return LightweightCNN(1, num_classes)

def create_model_single_channel(num_classes):
    return LightweightCNN(1, num_classes)

class CustomCombineChannel(object):
    def __init__(self, power_r, power_g, power_b):
        self.power_r = power_r
        self.power_g = power_g
        self.power_b = power_b

    def __call__(self, img):
        # img shape is [3, H, W] after ToTensor() in the range [0, 1]
        img_255 = img * 255.0
        r = img_255[0]
        g = img_255[1]
        b = img_255[2]
        
        combined = (r ** self.power_r) + (g ** self.power_g) + (b ** self.power_b)
        log_combined = torch.log(combined + 1e-7)
        normalized = log_combined / torch.log(torch.tensor(16646655.0))
        return normalized.unsqueeze(0)

class ExtractChannel(object):
    def __init__(self, channel):
        self.channel = channel
    def __call__(self, img):
        return img[self.channel:self.channel+1, :, :]

def create_dataloaders(transform, batch_size, num_workers):
    data_dir = 'data'
    cifar_path = os.path.join(data_dir, 'cifar-10-batches-py')
    needs_download = not os.path.exists(cifar_path)
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=needs_download, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=needs_download, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True)
    
    return train_loader, test_loader, train_dataset.classes

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            running_loss += loss.item() * X.size(0)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total

def run_training(model, train_loader, test_loader, log_dir, device, image_size=32, epochs=5):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir=log_dir)
    try:
        dummy = torch.randn(1, *([3, image_size, image_size] if next(model.parameters()).shape[1]==3 else [1, image_size, image_size])).to(device)
        writer.add_graph(model, dummy)
    except Exception as e:
        pass
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
    writer.close()
    return model, test_acc


class UltraTinyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)