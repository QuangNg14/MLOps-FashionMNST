import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io

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


# Model serving with Flask
app = Flask(__name__)
model = StylishNN().to(device)


def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return transform(image).unsqueeze(0).to(device)


@app.route("/", methods=["GET"])
def hello():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        return jsonify({"prediction": predicted.item()})


if __name__ == "__main__":
    model_path = "models/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    app.run(debug=True)
