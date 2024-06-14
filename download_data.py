# download_data.py
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(
    "./data", train=True, download=True, transform=transform
)
valid_dataset = datasets.FashionMNIST(
    "./data", train=False, download=True, transform=transform
)
