import os
import random
import numpy as np
import torch
import copy

from torch import device, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

def create_model_single_channel(num_classes):
    return LightweightCNN(1, num_classes)

class LinearCombineChannel(object):
    def __init__(self, w_r, w_g, w_b):
        self.w_r = w_r
        self.w_g = w_g
        self.w_b = w_b

    def __call__(self, img):
        r = img[0]
        g = img[1]
        b = img[2]
        combined = self.w_r * r + self.w_g * g + self.w_b * b
        return combined.unsqueeze(0)

def create_dataloaders(transform, batch_size, num_workers, subset_per_class=None):
    data_dir = 'data'
    cifar_path = os.path.join(data_dir, 'cifar-10-batches-py')
    needs_download = not os.path.exists(cifar_path)
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=needs_download, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=needs_download, transform=transform)
    
    if subset_per_class is not None:
        indices = []
        targets = np.array(train_dataset.targets)
        for i in range(len(train_dataset.classes)):
            class_indices = np.where(targets == i)[0][:subset_per_class]
            indices.extend(class_indices)
        train_dataset = Subset(train_dataset, indices)
        
        # Also subset test set for maximum speed during evaluation
        test_indices = []
        test_targets = np.array(test_dataset.targets)
        for i in range(len(test_dataset.classes)):
            test_indices.extend(np.where(test_targets == i)[0][:subset_per_class])
        test_dataset = Subset(test_dataset, test_indices)
    
    pw = True if num_workers > 0 else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=pw)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=pw)
    
    classes = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
    return train_loader, test_loader, classes


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        preds = model(X)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

        predicted = preds.argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)
            loss = loss_fn(preds, y)

            running_loss += loss.item() * X.size(0)

            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def run_training(
    model,
    train_loader,
    test_loader,
    log_dir,
    device,
    image_size=32,
    epochs=10,
    patience=2,
):
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    writer = SummaryWriter(log_dir=log_dir)

    try:
        dummy = torch.randn(1, 3, image_size, image_size).to(device)
        writer.add_graph(model, dummy)
    except Exception:
        pass

    best_acc = 0.0
    epochs_without_improvement = 0

    test_loss = float("nan")
    test_acc = float("nan")

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)

        should_eval = (
            (epoch + 1) % 3 == 0
            or epoch == epochs - 1
        )

        if should_eval:

            test_loss, test_acc = evaluate(
                model,
                test_loader,
                loss_fn,
                device,
            )

            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/acc", test_acc, epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {patience} "
                    f"evaluation rounds without improvement."
                )
                break

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Acc: {train_acc:.4f}")

        if should_eval:
            print(f"Validation Acc: {test_acc:.4f}")
    
    torch.save(
        model.state_dict(),
        os.path.join(log_dir, "best_model.pt")
    )
    
    writer.close() 
    return model, best_acc