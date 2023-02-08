# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torchvision.io import read_image

import torchvision
import torchvision.models as models
import torch
import os

from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

data_loc = '/kaggle/input'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_data_path = "traffic_data/Train"
test_data_path = "traffic_data/Test"

dataset = {
    "train" : torchvision.datasets.ImageFolder(root = train_data_path, transform =data_transforms["train"]),
    
    "test" : torchvision.datasets.ImageFolder(root = train_data_path, transform =data_transforms["test"])
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_loader ={
    "train" : DataLoader(dataset["train"], batch_size=64, shuffle=True),
    
    "test" : DataLoader(dataset["test"], batch_size=64, shuffle=True)
}

train_data="traffic_data/Train"
class_ids = ["Speed_Limit_20","Speed_Limit_30","Speed_Limit_50","Speed_Limit_60", "Speed_Limit_70", "Speed_Limit_80",
            "End_Speed_Limit_80","Speed_Limit_100","Speed_Limit_120","No_Passing","No_Passing_Over_3.5m","Intersection",
            "Priority_Road","Yield","Stop","No_Vehicles","Vehicles_over_3.5_prohibited","No_Entry","General_Caution",
            "Curve_Left","Curve_Right","Double_Curve","Bumpy_Road", "Slippery_road","Narrowroad_Right","Road_Work","Traffic_Lights",
            "Pedestrains","Children_Crossing","Bicycles_Crossing","Beware_Ice_and_Snow","Animals_Crossing","End_of_all_speed_limits",
            "Turn_Right","Turn_Left","Ahead_Only","Go_Straight_Right","Go_Straight_Left","Keep_Right","Keep_Left","Roundabout_Mandatory",
            "End_no_passing","End_no_passing_over_3.5"]

class_names_val={int(x) : class_ids[int(x)] for x in os.listdir(train_data)}

print(class_names_val)

dataset_sizes = {'train' : 50000, 'test': 10000}
train_features, train_labels = next(iter(dataset_loader["train"]))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = img.swapaxes(0, 1)
img = img.swapaxes(1, 2)
label = train_labels[0]
print(label)
print("Label: ", class_names_val[train_labels[0].item()])
plt.imshow(img)
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
import os
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataset_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataset_loader['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names_val[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

model_ft = models.wide_resnet50_2(pretrained=True)
#wideresnet.eval()
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 43)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

visualize_model(model_ft)