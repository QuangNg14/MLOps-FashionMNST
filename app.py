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
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Model serving with Flask
app = Flask(__name__)
model = StylishNN()


def transform_image(image_bytes):
    transform = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        tensor = tensor.to(device)
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        return jsonify({"prediction": predicted.item()})


if __name__ == "__main__":
    model_path = "models/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    app.run(host="0.0.0.0", port=5000)
