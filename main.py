#will use pytorch framework

#import required torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

import cv2
import models
from torch.autograd import Variable

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(torch.device)
#hyper params, dependant on data loaded

num_epochs = 0

batch_size = 4
learning_rate =0.001

#DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = {
    "test" : torchvision.datasets.CIFAR10(root='./data', train=False,
    transform=transform,download=True),

    "train" : torchvision.datasets.CIFAR10(root='./data', train=True,
    transform=transform, download=True)

}

dataset_loader ={
    "train" : torch.utils.data.DataLoader(dataset=dataset["train"], batch_size=batch_size, shuffle=True),

    "test" : torch.utils.data.DataLoader(dataset=dataset["test"], batch_size=batch_size, shuffle=False)
}
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#See sample data

examples =iter(dataset_loader["train"])
samples, labels = examples.next()

imshow(torchvision.utils.make_grid(samples))
#configure the network

model = models.CNN().to(device) #simple CNN model

#loss and optimiser

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training_loop

n_total_steps = len(dataset_loader["train"])
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataset_loader["train"]):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in dataset_loader["test"]:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
    
params = list(model.parameters()).to
weight = np.squeeze(params[-1].data.numpy())

examples =iter(dataset_loader["train"])
samples, labels = examples.next()

img = PILImage.create(get_x(sample.loc[0]))
img.show(figsize=(10,10));
