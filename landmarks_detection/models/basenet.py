import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .resnet import *

class Res152(nn.Module):
    def __init__(self,num_classes):
        super(Res152,self).__init__()
        self.pretrain_net = resnet152(pretrained=False)
        modules = list(self.pretrain_net.children())[:-1]
        self.base_net = nn.Sequential(*modules)
        self.fc = nn.Linear(8192,num_classes)

    def forward(self,x):
        x = self.base_net(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

if __name__=='__main__':
    model = Res152(6)
    summary(model, (3, 512, 512))