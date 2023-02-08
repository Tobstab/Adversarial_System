#main system implementation
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import torch

import torchvision
import torchvision.transforms as transforms
 
wideresnet = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
wideresnet.eval()