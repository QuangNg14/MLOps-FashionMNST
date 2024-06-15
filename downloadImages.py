import os
from torchvision import datasets
from PIL import Image

# Define the directory to save the images
save_dir = "fashion_mnist_images"
train_dir = os.path.join(save_dir, "train")
test_dir = os.path.join(save_dir, "test")

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Download the FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True)


# Function to save images
def save_images(dataset, directory):
    for idx, (image, label) in enumerate(dataset):
        label_dir = os.path.join(directory, str(label))
        os.makedirs(label_dir, exist_ok=True)
        image_path = os.path.join(label_dir, f"{idx}.png")
        image.save(image_path)


# Save training images
print("Saving training images...")
save_images(train_dataset, train_dir)

# Save test images
print("Saving test images...")
save_images(test_dataset, test_dir)

print("Images saved successfully!")
