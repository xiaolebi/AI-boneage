import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .SE_inceptionv3 import SEInception_v3
from .vit import ViT
from .VIT.modeling import VisionTransformer
from .InceptionV3 import Inception_v3
from .InceptionV3_SE_PAM_CAM import Inception_v3_SE_PAM
from .InceptionV3_PAM_CAM import Inception_v3_PAM

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

class BoneAge_InceptionV3_NO_SE(nn.Module):
    def __init__(self,num_class = 1):
        super(BoneAge_InceptionV3_NO_SE,self).__init__()
        pretrain_net = Inception_v3()
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
    
class BoneAge_InceptionV3_SE_PAM(nn.Module):
    def __init__(self,num_class = 1):
        super(BoneAge_InceptionV3_SE_PAM,self).__init__()
        pretrain_net = Inception_v3_SE_PAM()
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
    
class BoneAge_InceptionV3_PAM(nn.Module):
    def __init__(self,num_class = 1):
        super(BoneAge_InceptionV3_PAM,self).__init__()
        pretrain_net = Inception_v3_PAM()
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
 
class BoneAge_inception_vit(nn.Module):
    def __init__(self,image_size=14, patch_size=2, num_classes=1024, dim=128, depth=12, heads=8, mlp_dim=1000,channels = 2048):
        super(BoneAge_inception_vit,self).__init__()
        backbone = SEInception_v3()
        pretrain_net = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels)
        self.base_net = backbone
        self.next_net = pretrain_net
        self.gender_dense = nn.Linear(1,32)
        self.fc1 = nn.Linear(1024 + 1*32,512)
        self.last_layer = nn.Linear(512,1)

    def forward(self,x,gender_input):
        x = self.base_net(x)
        x = self.next_net(x)
        gender_dense = self.gender_dense(gender_input)
        x = torch.cat((x,gender_dense),dim=-1)
        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x,p=0.5,training=self.training)
        last_output = self.last_layer(x)
        return last_output

class BoneAge_vit(nn.Module):
    def __init__(self,image_size=512, patch_size=64, num_classes=1024, dim=128, depth=12, heads=8, mlp_dim=1000):
        super(BoneAge_vit,self).__init__()
        pretrain_net = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)
        self.base_net = pretrain_net
        self.gender_dense = nn.Linear(1,32)
        self.fc1 = nn.Linear(1024 + 1*32,512)
        self.last_layer = nn.Linear(512,1)

    def forward(self,x,gender_input):
        x = self.base_net(x)
        gender_dense = self.gender_dense(gender_input)
        x = torch.cat((x,gender_dense),dim=-1)
        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x,p=0.5,training=self.training)
        last_output = self.last_layer(x)
        return last_output
    
class BoneAge_VisionTransformer(nn.Module):
    def __init__(self,config,img_size=512, num_classes=1000, zero_head=False, vis=False,pretrain=True,weight=None):
        super(BoneAge_VisionTransformer, self).__init__()
        net = VisionTransformer(config,img_size=img_size)
        if pretrain:
            pretrain_weight = np.load(weight)
            net.load_from(pretrain_weight)
        self.base_net = net
        self.last_layer = nn.Linear(512,1)
        self.gender_dense = nn.Linear(1,32)
        self.fc1 = nn.Linear(1000 + 1*32,512)
        nn.init.xavier_uniform_(self.last_layer.weight)
        nn.init.normal_(self.last_layer.bias, std=1e-6)
        nn.init.xavier_uniform_(self.gender_dense.weight)
        nn.init.normal_(self.gender_dense.bias, std=1e-6)
        
    def forward(self, x,gender_input):
        x = self.base_net(x)
        gender_dense = self.gender_dense(gender_input)
        x = torch.cat((x[0],gender_dense),dim=-1)
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
