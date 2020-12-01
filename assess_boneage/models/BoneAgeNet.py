import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .SE_inceptionv3 import SEInception_v3

class BoneAge(nn.Module):
    def __init__(self,num_class = 1):
        super(BoneAge,self).__init__()
        pretrain_net = SEInception_v3()
        self.base_net = pretrain_net
        self.gender_dense = nn.Linear(1,32)
        self.fc1 = nn.Linear(2048*4 + 1*32,1000)
        self.last_layer = nn.Linear(1000,num_class)

    def forward(self,x,gender_input):
        x = self.base_net(x)
        gender_dense = self.gender_dense(gender_input)
        x = torch.cat((x,gender_dense),dim=-1)
        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x,p=0.5,training=self.training)
        last_output = self.last_layer(x)
        return last_output

class summary_model(nn.Module):
    def __init__(self,num_class = 1):
        super(summary_model,self).__init__()
        pretrain_net = SEInception_v3()
        self.base_net = pretrain_net
        self.gender_dense = nn.Linear(1,32)
        self.fc1 = nn.Linear(8192,1000)
        self.last_layer = nn.Linear(1000,num_class)

    def forward(self,x):
        x = self.base_net(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x,p=0.5,training=self.training)
        last_output = self.last_layer(x)
        return last_output

if __name__ == '__main__':
    model = summary_model(1)
    summary(model,(3,512,512))
