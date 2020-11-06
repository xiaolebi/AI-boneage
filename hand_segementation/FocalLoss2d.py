import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss2d(nn.Module):
    def __init__(self,weight=None,alpha=0,size_average=True):
        super(FocalLoss2d, self).__init__()
        if weight is None:
            pass
        else:
            if isinstance(weight,Variable):
                self.weight = weight
            else:
                self.weight = Variable(weight)
        self.alpha = alpha
        self.size_average = size_average

    def forward(self,inputs,targets):
        batch_size,class_size,width_size,height_size=inputs.size()
        P = F.softmax(inputs)
        # print('softmax\n',P)
        if self.weight is None:
            self.weight = Variable(torch.ones(class_size))
        ground_truth = inputs.data.new(batch_size,class_size,width_size,height_size).fill_(0)
        ids = targets.view(batch_size,1,width_size,height_size)
        # print('ids\n',ids)
        ground_truth.scatter_(1,ids.data,1.)
        # print('groundtruth\n',ground_truth)
        ground_truth = Variable(ground_truth)
        if inputs.is_cuda and not self.weight.is_cuda:
            self.weight = self.weight.cuda
        weight = self.weight[ids.data.view(-1)].view(batch_size,width_size,height_size)
        probs = (P*ground_truth).sum(1).view(batch_size,width_size,height_size)
        # print('probs\n',probs)
        log_p = probs.log()
        batch_loss = -weight*(torch.pow((1-probs),self.alpha))*log_p
        # print('batchloss\n', batch_loss)
        batch_loss = batch_loss.view(batch_size,-1).sum(-1)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

if __name__ == '__main__':
    input_data = [[[[2,3],[1,3]],[[1,4],[2,4]]]]
    input_data = torch.autograd.Variable(torch.FloatTensor(input_data))

    label = [[[0,1],[1,1]]]
    label = torch.autograd.Variable(torch.LongTensor(label))
    criterion = FocalLoss2d(torch.FloatTensor([6,4]),2)
    loss = criterion(input_data,label)
    print(loss.data.tolist())
