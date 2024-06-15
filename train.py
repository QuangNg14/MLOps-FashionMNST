import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import subprocess


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model definition
class StylishNN(nn.Module):
    def __init__(self):
        super(StylishNN, self).__init__()
        self.l1 = nn.Conv2d(1, 12, (5, 5), stride=1, padding="same")
        self.a1 = nn.ReLU()
        self.m1 = nn.MaxPool2d(2, stride=2)
        self.l2 = nn.Conv2d(12, 32, (5, 5), stride=1, padding=(1, 1))
        self.a2 = nn.ReLU()
        self.m2 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6 * 6 * 32, 600)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.m1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.m2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x


# Training function
def train(model, train_loader, opt, epoch, verbose=False):
    if verbose:
        print("starting epoch", epoch)
    train_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    for i, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)  # Uses GPUs if possible

        # Forward pass
        ypred_batch = model(image)

        # Calculate loss
        loss = loss_fn(ypred_batch, label)

        # Backward pass
        loss.backward()

        # Update weight estimates
        opt.step()

        # Reset gradients to zero
        opt.zero_grad()

        # Track the sum of losses to calculate the average training loss for this epoch
        train_loss += loss.item()

        # Verbose logging
        if verbose and ((i % 100) == 0):
            print(
                "training [epoch {}: {}/{} ({:.0f}%)] loss: {:.6f}".format(
                    epoch,
                    i * len(image),
                    len(train_loader.dataset),
                    100.0 * i / len(train_loader),
                    loss.item(),
                )
            )

    avg_tl = train_loss / (i + 1)
    print("epoch {} avg training loss: {:.6f}".format(epoch, avg_tl))
    return avg_tl


# Validation function
def valid(model, valid_loader):
    valid_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    list_predictions_valid = []
    true_labels = []

    with torch.no_grad():  # For validation, we do not need to update gradients
        for i, (image, label) in enumerate(valid_loader):
            image, label = image.to(device), label.to(device)

            # Get the model prediction
            pred = model(image)

            # Update our running tally of the validation loss
            valid_loss += loss_fn(pred, label).item()

            # Calculate the accuracy for this batch
            _, pred = torch.max(pred.data, 1)
            correct += torch.sum(label == pred).item()

            np_pred = pred.cpu().detach().numpy()
            np_true_labels = label.cpu().detach().numpy()
            for num in np_pred:
                list_predictions_valid.append(num)
            for num in np_true_labels:
                true_labels.append(num)

    avg_vl = valid_loss / (i + 1)
    print(
        "avg validation loss: {:.6f}, accuracy: {}/{} (({:.0f}%))".format(
            avg_vl,
            correct,
            len(valid_loader.dataset),
            100.0 * correct / len(valid_loader.dataset),
        )
    )

    return avg_vl


# Data loading
# def load_data():
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = datasets.FashionMNIST(
#         "./data", train=True, download=True, transform=transform
#     )
#     valid_dataset = datasets.FashionMNIST(
#         "./data", train=False, download=True, transform=transform
#     )
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
#     return train_loader, valid_loader


# Data loading
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    data_path = "./data"

    # Ensure data directory exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        # Pull the dataset using DVC if not already available
        subprocess.run(["dvc", "pull", "data/FashionMNIST.dvc"])

    train_dataset = datasets.FashionMNIST(
        data_path, train=True, download=False, transform=transform
    )
    valid_dataset = datasets.FashionMNIST(
        data_path, train=False, download=False, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    return train_loader, valid_loader


# Training and validation process
def run_training():
    train_loader, valid_loader = load_data()
    model = StylishNN().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, opt, epoch, verbose=True)
        valid_loss = valid(model, valid_loader)

    # Save the best model
    model_path = "models/best_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # DVC commands to track the model
    subprocess.run(["dvc", "add", model_path])
    subprocess.run(["git", "add", "models/best_model.pth.dvc"])
    subprocess.run(["git", "commit", "-m", "Update model"])
    subprocess.run(["dvc", "push"])


if __name__ == "__main__":
    run_training()
