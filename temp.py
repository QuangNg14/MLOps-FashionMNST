# import numpy as np
# import pandas as pd
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# print("device: " + device)

# fm_data_train = torchvision.datasets.FashionMNIST(
#     "./data", download=True, transform=transforms.Compose([transforms.ToTensor()])
# )
# fm_data_valid = torchvision.datasets.FashionMNIST(
#     "./data",
#     download=True,
#     train=False,
#     transform=transforms.Compose([transforms.ToTensor()]),
# )


# # based on the label definitions given by the original dataset
# # https://github.com/zalandoresearch/fashion-mnist#labels
# # we have provided a convenient function to get the actual labels
# # corresponding to each numeric label in the dataset
# def label_name(label):
#     label_mapping = {
#         0: "T-shirt/Top",
#         1: "Trouser",
#         2: "Pullover",
#         3: "Dress",
#         4: "Coat",
#         5: "Sandal",
#         6: "Shirt",
#         7: "Sneaker",
#         8: "Bag",
#         9: "Ankle Boot",
#     }
#     label_num = label.item() if type(label) == torch.Tensor else label
#     return label_mapping[label_num]


# label_mapping = {
#     0: "T-shirt/Top",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }

# tens_to_img = transforms.ToPILImage()
# for i in range(4):
#     print("label id:", fm_data_train[i][1], "name:", label_name(fm_data_train[i][1]))
#     display(tens_to_img(fm_data_train[i][0]).resize((150, 150)))

# # TODO: how many observations are there in your training and validation sets?
# print(len(fm_data_train))
# print(len(fm_data_valid))

# # TODO: make data loaders for the training and validation datasets
# # that we loaded above; for now, start with a batch size of 1
# batch_size = 1
# from torch.utils.data import DataLoader

# train_loader = DataLoader(fm_data_train, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(fm_data_valid, batch_size=batch_size, shuffle=True)

# # TODO: loop through the entire dataset using the DataLoaders you constructed earlier
# # and count how many observations we have per label (in both training and validation sets)
# # show the results using the label name (not simply the number representation)
# from collections import defaultdict

# mapp = defaultdict(int)
# for i, batch in enumerate(train_loader):
#     # print(f'batch {i}:', 'x:', batch[0], 'y:', batch[1])
#     mapp[label_name(batch[1])] += 1

# for i, batch in enumerate(valid_loader):
#     # print(f'batch {i}:', 'x:', batch[0], 'y:', batch[1])
#     mapp[label_name(batch[1])] += 1
# print(mapp)


# # first model attempt
# class StylishNN(nn.Module):
#     def __init__(self, num_classes=10):
#         # TODO: inherit from nn.Module
#         super().__init__()  # inherit from nn.Module
#         # input: 28x28x1, output: 28x28x12

#         # TODO: add a first convolution layer together with a ReLU activation function
#         # for this first convolution layer, we want to apply 12 filters, where each
#         # filter is 5x5, taking strides of 1, and padded to maintain the original image size
#         # follow up on this convolution with max pooling with a 2x2 filter and stride 2
#         self.l1 = nn.Conv2d(1, 12, (5, 5), stride=1, padding="same")  # padding of 2
#         self.a1 = nn.ReLU()  # activation
#         self.m1 = nn.MaxPool2d(2, stride=2)  # 14x14x12

#         # TODO: add a second convolution layer that expands the number of filters to 32
#         # the filters will still be 5x5 and taking strides of 1, but this time, use a 1 pixel padding instead
#         # follow up on this convolution with max pooling again with a 2x2 filter and stride 2

#         self.l2 = nn.Conv2d(12, 32, (5, 5), stride=1, padding=(1, 1))  # 12x12x32
#         self.a2 = nn.ReLU()  # activation
#         self.m2 = nn.MaxPool2d(2, stride=2)  # 6x6x32
#         # TODO: for the classification portion of this neural network, make 3 fully connected layers
#         # you will need to calculate what the current dimensions of your tensors are after convolution
#         # and pooling, as you the fully connected layers will use the flattened values
#         # with that number as your initial input, here are the number of neurons for each layer, together
#         # with which activation function we would like to use

#         # fully connected layer 1: 600 neurons, ReLU activation
#         # fully connected layer 2: 120 neurons, ReLU activation
#         # fully connected layer 3 (output layer): num_classes neurons, no additional activation function,
#         # use the log(softmax) of these values to correspond to a probability for each class
#         self.flatten = nn.Flatten()
#         # self.dropout1 = nn.Dropout2d()
#         # First fully connected layer
#         self.fc1 = nn.Linear(6 * 6 * 32, 600)
#         # Second fully connected layer
#         self.fc2 = nn.Linear(600, 120)
#         # third fully connected layer that outputs our 10 labels
#         self.fc3 = nn.Linear(120, num_classes)
#         # log softmax
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         # TODO: forward pass for the convolution and pooling layers
#         x = self.l1(x)
#         x = self.a1(x)
#         x = self.m1(x)
#         x = self.l2(x)
#         x = self.a2(x)
#         x = self.m2(x)
#         # TODO: flatten parameters into 1 dimension before passing them
#         # on to our fully connected layers
#         x = self.flatten(x)
#         # x = self.dropout1(x)
#         # TODO: forward pass for the fully connected layers
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         # TODO: return the log(softmax) predictions for each class
#         x = self.log_softmax(x)
#         return x

#     # TODO: make data loaders for the training and validation datasets


# # with a batch size of 64; remember that we want to randomize the order that we see our data!
# batch_size = 64

# train_loader = DataLoader(fm_data_train, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(fm_data_valid, batch_size=batch_size, shuffle=True)

# # TODO: fill in the steps that we want to take within the training loop
# # (this function will run once per epoch)
# # we have provided some helper code for logging progress


# def train(model, train_loader, opt, epoch, verbose=False):
#     if verbose:
#         print("starting epoch", epoch)
#     train_loss = 0
#     l2_lambda = 1
#     loss_fn = nn.CrossEntropyLoss()

#     for i, (image, label) in enumerate(train_loader):
#         image, label = image.to(device), label.to(device)  # uses GPUs if possible

#         # TODO: a) forward pass
#         # ypred_batch = model(image)[:, 0]
#         ypred_batch = model(image)

#         # TODO: b) calculate loss
#         loss = loss_fn(ypred_batch, label)

#         # TODO: c) backward pass
#         loss.backward()
#         # TODO: d) update weight estimates
#         opt.step()
#         # TODO: e) reset gradients to zero
#         opt.zero_grad()

#         # we are tracking the sum of losses to calculate the average training loss for this epoch
#         train_loss += loss.item()

#         # when verbose is on, we will show how the loss is changing across batches within one epoch
#         if verbose and ((i % 100) == 0):
#             print(
#                 "training [epoch {}: {}/{} ({:.0f}%)] loss: {:.6f}".format(
#                     epoch,
#                     i * len(image),
#                     len(train_loader.dataset),
#                     100.0 * i / len(train_loader),
#                     loss.item(),
#                 )
#             )

#     avg_tl = train_loss / (i + 1)
#     print("epoch {} avg training loss: {:.6f}".format(epoch, avg_tl))
#     return avg_tl


# # we have provided the below code for reporting loss and accuracy
# # on your validation set, so simply run this snippet without any changes
# list_predictions_valid = []
# true_labels = []


# def valid(model, valid_loader):
#     valid_loss = 0
#     correct = 0
#     loss_fn = nn.CrossEntropyLoss()
#     with torch.no_grad():  # for validation, we do not need to update gradients
#         for i, (image, label) in enumerate(valid_loader):
#             image, label = image.to(device), label.to(device)

#             # get the model prediction
#             pred = model(image)

#             # update our running tally of the validation loss
#             valid_loss += loss_fn(pred, label).item()

#             # calculate the accuracy for this batch
#             _, pred = torch.max(pred.data, 1)
#             correct += torch.sum(label == pred).item()
#             np_pred = pred.cpu().detach().numpy()
#             # Get the predicted class with the highest score.
#             # Compare the predicted classes with the true labels and count the number of correct predictions.
#             # Accumulate the correct predictions count.
#             np_true_labels = label.cpu().detach().numpy()
#             for num in np_pred:
#                 list_predictions_valid.append(num)
#             for num in np_true_labels:
#                 true_labels.append(num)

#     # get the loss for the epoch
#     avg_vl = valid_loss / (i + 1)
#     print(
#         "avg validation loss: {:.6f}, accuracy: {}/{} (({:.0f}%))".format(
#             avg_vl,
#             correct,
#             len(valid_loader.dataset),
#             100.0 * correct / len(valid_loader.dataset),
#         )
#     )

#     return avg_vl


# # we can now initialize the model you designed above
# # and optionally put it on the GPU if we have enabled it
# # remember to rerun this if you want to re-initialize your parameters!
# # (otherwise, the model will simply keep updating the previous parameters)
# model = StylishNN().to(device)
# # TODO: initialize an SGD optimizer with learning rate 0.01
# opt = torch.optim.SGD(model.parameters(), lr=0.1)
# # TODO: initialize cross entropy as your loss function
# loss = nn.CrossEntropyLoss()

# # now that you have defined your model and set up the training loop
# # we can run through 15 epochs and see how our training and validation
# # losses change over time (you can just run this code directly)
# # note that we have turned on verbose here to better see intermediate progress
# # but if the output is overwhelming, you can feel free to turn it off
# epoch_list = []
# train_loss = []
# valid_loss = []

# epochs = 15
# for e in range(1, epochs + 1):
#     epoch_list.append(e)
#     train_loss.append(train(model, train_loader, opt, e, verbose=True))
#     valid_loss.append(valid(model, valid_loader))

#     # TODO: as we trained our model, we captured the epoch,
# # average loss for the training and validation sets in
# # epoch_list, train_loss, and valid_loss
# # use these to plot the loss vs epochs for the training and test set

# plt.plot(epoch_list, train_loss, label="Training Loss")
# plt.plot(epoch_list, valid_loss, label="Validation Loss")
# plt.title("Average Loss for training vs validation sets")
# plt.legend()
# plt.show()

# len(list_predictions_valid)

# # TODO: use sklearn's convenient confusionmatrix and ConfusionMatrixDisplay
# # methods to calculate and visualize the confusion matrix for your
# # current predictions, making sure to display the human-readable
# # class labels (instead of 0-9)
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# plt.figure(figsize=(30, 30))
# cm = confusion_matrix(
#     true_labels, list_predictions_valid, labels=list(label_mapping.keys())
# )
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm, display_labels=list(label_mapping.values())
# )
# disp.plot()
# plt.show()

import mlflow
import mlflow.pytorch


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


def run_training():
    logger.info("Starting training...")
    train_loader, valid_loader = load_data()
    model = StylishNN().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 10
    with mlflow.start_run() as run:
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, opt, epoch, verbose=True)
            valid_loss = valid(model, valid_loader)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)

        # Log the model
        mlflow.pytorch.log_model(model, "model")

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
    logger.info("Training completed.")


if __name__ == "__main__":
    run_training()
