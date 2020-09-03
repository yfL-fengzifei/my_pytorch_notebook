import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torchvision

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
