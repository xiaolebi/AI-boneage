from .se_module import SELayer
from torchsummary import summary
from torchvision.models.inception import Inception3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .da_att import PAM_Module,CAM_Module

__all__ = ['Inception3','Inception_v3_SE_PAM']

model_urls = {
    'inception_v3_google':'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
}


'''
This model is inceptionV3+SENet+PAM_Module/CAM_Module
'''
def Inception_v3_SE_PAM(pretrained=False,**kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model
    return Inception3(**kwargs)

class Inception3(nn.Module):
    def __init__(self,num_classes=1000,aux_logits=True,transform_input=False):
        super(Inception3,self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3,32,kernel_size=3,stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32,32,kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3,padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Conv2d_get_feature = nn.Conv2d(2048,3,kernel_size=1,bias=False)
        self.Mixed_5b = nn.Sequential(
            InceptionA(192,pool_features=32),
            SELayer(256))
        self.PAM_1 = PAM_Module(256)
        self.CAM_1 = CAM_Module(256)

        self.Mixed_5c = nn.Sequential(
            InceptionA(256, pool_features=64),
            SELayer(288))
        self.PAM_2 = PAM_Module(288)
        self.CAM_2 = CAM_Module(288)

        self.Mixed_5d = nn.Sequential(
            InceptionA(288, pool_features=64),
            SELayer(288))
        self.PAM_3 = PAM_Module(288)
        self.CAM_3 = CAM_Module(288)
        # downsample
        self.Mixed_6a = nn.Sequential(
            InceptionB(288),
            SELayer(768))
        self.PAM_4 = PAM_Module(768)
        self.CAM_4 = CAM_Module(768)

        self.Mixed_6b = nn.Sequential(
            InceptionC(768, channels_7x7=128),
            SELayer(768))
        self.PAM_5 = PAM_Module(768)
        self.CAM_5 = CAM_Module(768)

        self.Mixed_6c = nn.Sequential(
            InceptionC(768, channels_7x7=160),
            SELayer(768))
        self.PAM_6 = PAM_Module(768)
        self.CAM_6 = CAM_Module(768)

        self.Mixed_6d = nn.Sequential(
            InceptionC(768, channels_7x7=160),
            SELayer(768))
        self.PAM_7 = PAM_Module(768)
        self.CAM_7 = CAM_Module(768)

        self.Mixed_6e = nn.Sequential(
            InceptionC(768, channels_7x7=192),
            SELayer(768))
        self.PAM_8 = PAM_Module(768)
        self.CAM_8 = CAM_Module(768)

        if aux_logits:
            self.AuxLogits = InceptionAux(768,num_classes)
        # downsample
        self.Mixed_7a = nn.Sequential(
            InceptionD(768),
            SELayer(1280))
        self.PAM_9 = PAM_Module(1280)
        self.CAM_9 = CAM_Module(1280)

        self.Mixed_7b = nn.Sequential(
            InceptionE(1280),
            SELayer(2048))
        self.PAM_10 = PAM_Module(2048)
        self.CAM_10 = CAM_Module(2048)

        self.Mixed_7c = nn.Sequential(
            InceptionE(2048),
            SELayer(2048))
        self.PAM_11 = PAM_Module(2048)
        self.CAM_11 = CAM_Module(2048)

        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m,'stddev') else 0.1
                X = stats.truncnorm(-2,2,scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.data.numel()),dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.data.copy_(values)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x,kernel_size=3,stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x,kernel_size=3,stride=2)

        x = self.Mixed_5b(x)
        x1 = self.PAM_1(x)
        x2 = self.CAM_1(x)
        x = x1 + x2

        x = self.Mixed_5c(x)
        x1 = self.PAM_2(x)
        x2 = self.CAM_2(x)
        x = x1 + x2

        x = self.Mixed_5d(x)
        x1 = self.PAM_3(x)
        x2 = self.CAM_3(x)
        x = x1 + x2

        x = self.Mixed_6a(x)
        x1 = self.PAM_4(x)
        x2 = self.CAM_4(x)
        x = x1 + x2
        # """In practice, we have found that employing this factorization does not
        # work well on early layers, but it gives very good results on medium
        # grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        # On that level, very good results can be achieved by using 1 × 7 convolutions
        # followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x)
        x1 = self.PAM_5(x)
        x2 = self.CAM_5(x)
        x = x1 + x2

        x = self.Mixed_6c(x)
        x1 = self.PAM_6(x)
        x2 = self.CAM_6(x)
        x = x1 + x2

        x = self.Mixed_6d(x)
        x1 = self.PAM_7(x)
        x2 = self.CAM_7(x)
        x = x1 + x2

        x = self.Mixed_6e(x)
        x1 = self.PAM_8(x)
        x2 = self.CAM_8(x)
        x = x1 + x2

        # Efficient Grid Size Reduction
        x = self.Mixed_7a(x)
        x1 = self.PAM_9(x)
        x2 = self.CAM_9(x)
        x = x1 + x2

        # We are using this solution only on the coarsest grid,
        # since that is the place where producing high dimensional
        # sparse representation is the most critical as the ratio of
        # local processing (by 1 × 1 convolutions) is increased compared
        # to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x1 = self.PAM_10(x)
        x2 = self.CAM_10(x)
        x = x1 + x2

        x = self.Mixed_7c(x)
        x1 = self.PAM_11(x)
        x2 = self.CAM_11(x)
        x = x1 + x2


        x = F.avg_pool2d(x,kernel_size=7)

        x = x.view(x.size(0), -1)
        return x


class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(input_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(input_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branchpool = BasicConv2d(input_channels, pool_features, kernel_size=1)

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branchpool(branch_pool)

        #x -> pool -> 1x1(same)
        outputs = [branch1x1,branch5x5,branch3x3dbl,branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(input_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):

        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        #x -> maxpool(downsample)
        branchpool = F.max_pool2d(x,kernel_size=3,stride=2)

        #"""We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3dbl, branchpool]

        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super(InceptionC,self).__init__()
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)

        c7 = channels_7x7
        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7_1 = BasicConv2d(input_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0,3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7,1), padding=(3,0))

        self.branch7x7dbl_1 = BasicConv2d(input_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 =BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 =BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 =BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 =BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))


        self.branch_pool = BasicConv2d(input_channels, 192, kernel_size=1)

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        #x-> avgpool (same)
        branchpool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool = self.branch_pool(branchpool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branchpool]

        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3_1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7_1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):

        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7 = self.branch7x7_4(branch7x7)

        #x -> avgpool (downsample)
        branchpool = F.max_pool2d(x,kernel_size=3,stride=2)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3stack_1 = BasicConv2d(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(input_channels, 192, kernel_size=1)

    def forward(self, x):

        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs.
        #This architecture is used on the coarsest (8 × 8) grids to promote
        #high dimensional representations, as suggested by principle
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool = self.branch_pool(branchpool)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionAux(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(InceptionAux,self).__init__()
        self.conv0 = BasicConv2d(in_channels,128,kernel_size=1)
        self.conv1 = BasicConv2d(128,768,kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768,num_classes)
        self.fc.stddev = 0.001

    def forward(self,x):
        #17x17x768
        x = F.avg_pool2d(x,kernel_size=5,stride=3)
        #5x5x768
        x = self.conv0(x)
        #5x5x128
        x = self.conv1(x)
        #1x1x768
        x = x.view(x.size(0),-1)
        #768
        x = self.fc(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels,eps=0.001)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x,inplace=True)