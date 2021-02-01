#修改注意：lr、checkpoints、resume、loggerpath、model_best path

from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import torch
import logging
import logging.handlers
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from models import *
from AgeDataset import *
from models.BoneAgeNet import BoneAge
from utils import AverageMeter,normalizedME,mkdir_p

parser = argparse.ArgumentParser(description='PyTorch hand landmark training')
parser.add_argument('--dataset',default='BoneageAssessmentDataset')
parser.add_argument('--workers',default=2,type=int,metavar='N',help='number of data loading workers (default:4)')
parser.add_argument('--epochs',default=120,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch',default=16,type=int,metavar='N',help='train batch size')
parser.add_argument('--test_batch',default=3,type=int,metavar='N',help='test batch size')
parser.add_argument('--lr','--learning-rate',default=0.001,type=float,metavar='LR',help='initial learning rate') #0.000063
parser.add_argument('--drop','--dropout',default=0,type=float,metavar='Dropout',help='Dropout ratio')
parser.add_argument('--schedule',type=int,nargs='+',default=[5,10,20,30,50,70],help='Decrease learning rate at these epochs')
parser.add_argument('--gamma',type=float,default=0.1,help='LR is multiplied by gamma on schedule')
parser.add_argument('--momentum',default=0.9,type=float,metavar='M',help='momentum')
parser.add_argument('--weight_decay','--wd',default=1e-4,type=float,metavar='W',help='weight decay (default: 1e-4)')
parser.add_argument('--panelty','--pl',default=1e-4,type=float)
parser.add_argument('--checkpoint',default='/content/checkpoints',type=str,metavar='PATH',help='path to save checkpoint(default:checkpoint)')
parser.add_argument('--resume',default='',type=str,metavar='PATH',help='path to latest checkpoint(default:None)')#/content/checkpoints/resume/Assess_BoneAge_InceptionV3_4.pth.tar
parser.add_argument('--depth',type=int,default=104,help='Model depth')
parser.add_argument('--cardinality',type=int,default=8,help='Model cardinality(group)')
parser.add_argument('--widen_factor',type=int,default=4,help='Widen factor 4 -> 64,8 -> 128')
parser.add_argument('--growthRate',type=int,default=12,help='Growth rate for DenseNet')
parser.add_argument('--compressionRate',type=int,default=2,help='Compression Rate(theta) for DenseNet')
parser.add_argument('--manualSeed',type=int,help='manual seed')
parser.add_argument('--e','--evaluate',dest='evaluate',action='store_true',help='evaluate model on validation set')
parser.add_argument('--gpu_id',default='0',type=str,help='id for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k:v for k,v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
if args.manualSeed is None:
    args.manualSeed = random.randint(1,10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
best_acc = 999
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler('/content/drive/My Drive/assess_boneage/assess_boneage/0201/20210201_16batch.log')
fmt = logging.Formatter('[%(asctime)s] - %(filename)s [Line:%(lineno)d] - [%(levelname)s] - %(message)s')
handler.setFormatter(fmt)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

def main():
    global best_acc
    start_epoch = args.start_epoch
    if not os.path.exists(args.checkpoint):
        mkdir_p(args.checkpoint)
    print("==> Preparing dataset %s"%args.dataset)
    transform_train = transforms.Compose([
        Rescale((520,520)),
        RandomCrop((512,512)),
        # RandomFlip(),
        # RandomBrightness(),
        ToTensor(512)
    ])
    transform_test = transforms.Compose([
        Rescale((512, 512)),
        ToTensor(512)
    ])

    trainset = AgeDataset(csv_file="/content/dataset/train.csv",transform=transform_train,root_dir='/content/dataset/train')
    trainloader = data.DataLoader(trainset,batch_size=args.train_batch,shuffle=True,num_workers=args.workers)
    testset = AgeDataset(csv_file='/content/dataset/valid.csv',transform=transform_test,root_dir='/content/dataset/valid')
    testloader = data.DataLoader(testset,batch_size=args.test_batch,shuffle=True,num_workers=args.workers)
    model = BoneAge(1)
    model.apply(weights_init)
    cudnn.benchmark = True
    print('   Total params: %.2fM'%(sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.SmoothL1Loss().cuda()
    ignored_params = []
    base_params = filter(lambda p:id(p) not in ignored_params,model.parameters())
    '''
    params = [
        {'params':base_params, 'lr':args.lr},
        {'params':model.fc.parameters(),'lr':args.lr*10}
    ]
    '''
    params = [
        {'params':base_params,'lr':args.lr  }
    ]
    
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(params=params,lr=args.lr,weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    title = 'Assess_BoneAge_InceptionV3'
    if args.resume:
        print('==> Resume from checkpoint...')
        logging.info('==> Resume from checkpoint %s'%(args.resume))
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss,test_acc = test(testloader,model,criterion,start_epoch,use_cuda)
        print(' Test Loss: %.8f, Test Acc: %.2f'%(test_loss,test_acc))
        return

    for epoch in range(start_epoch,args.epochs):
#         adjust_learning_rate(optimizer,epoch)
        print('\nEpoch: [%d | %d] LR: %f'%(epoch+1,args.epochs,state['lr']))
        train_loss,trian_acc = train(trainloader,model,criterion,optimizer,epoch,use_cuda,scheduler)
        test_loss,test_acc = test(testloader,model,criterion,start_epoch,use_cuda)
        print('\nLR:%f train_loss:%s trian_acc:%s test_loss:%s test_acc:%s' %(state['lr'],train_loss,trian_acc,test_loss,test_acc))
        is_best = test_loss<best_acc
        best_acc = min(test_loss,best_acc)
        logging.info('LR:%f epoch:%s train_loss:%s test_loss:%s best_loss:%s'%(state['lr'],epoch,train_loss,test_loss,best_acc))
        if (epoch+1)%5 == 0 or test_loss<6.0:
            save_checkpoint({
                'epoch':epoch+1,
                'state_dict':model.state_dict(),
                'test_acc':test_acc,
                'best_acc':best_acc,
                'optimizer':optimizer.state_dict(),
            },is_best,checkpoint=args.checkpoint,filename=title+'_'+str(epoch)+'.pth.tar')

    print('Best acc')
    print(best_acc)

def train(trainloader,model,criterion,optimizer,epoch,use_cuda,scheduler):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for batch_idx,batch_data in enumerate(trainloader):
        data_time.update(time.time()-end)
        inputs = batch_data['image']
        targets = batch_data['landmarks']
        gender = batch_data['gender']
        if use_cuda:
            inputs,targets,gender = inputs.cuda(),targets.cuda(),gender.cuda()
        inputs,targets,gender = torch.autograd.Variable(inputs),torch.autograd.Variable(targets),torch.autograd.Variable(gender)
        outputs = model(inputs,gender)
        loss = criterion(outputs,targets)
        '''
        l2_reg = torch.autograd.Variable(torch.cuda.FloatTensor(1),requires_grad=True)
        for w model.parameters():
            l2_reg = l2_reg + torch.pow(w,2).sum()
        '''
        l2_reg = Variable(torch.cuda.FloatTensor(1),requires_grad=True)
        #l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        for W in model.parameters():
            l2_reg = l2_reg + torch.pow(W,2).sum()
        loss = loss + args.panelty*l2_reg
        losses.update(loss.item(),inputs.size(0))
        optimizer.zero_grad()
        print("batch:{} train loss:{}".format(batch_idx,losses.avg)) 
        loss.backward()
        scheduler.step(loss)
        optimizer.step()
        state['lr'] = optimizer.param_groups[0]['lr']
        batch_time.update(time.time()-end)
        end = time.time()
        # print('Train:({batch}/{size}) Data:{data:.3f}s | Batch:{bt:.3f}s | Loss:{loss:.4f}'.format(
        #     batch = batch_idx,
        #     size = len(trainloader),
        #     data = data_time.avg,
        #     bt = batch_time.avg,
        #     loss = losses.avg,
        # ))
    return (losses.avg,0)

def test(testloader,model,criterion,epoch,use_cuda):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    for batch_idx,batch_data in enumerate(testloader):
        data_time.update(time.time()-end)
        inputs = batch_data['image']
        targets = batch_data['landmarks']
        gender = batch_data['gender']
        if use_cuda:
            inputs, targets, gender = inputs.cuda(), targets.cuda(), gender.cuda()
        inputs, targets, gender = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(gender)
        outputs = model(inputs,gender)
        loss = criterion(outputs,targets)
        losses.update(loss.item(),inputs.size(0))
        print("batch:{} test loss:{}".format(batch_idx,losses.avg))
        batch_time.update(time.time()-end)
        end = time.time()
        # print('Test:({batch}/{size}) Data:{data:.3f}s | Batch:{bt:.3f}s | Loss:{loss:.4f}'.format(
        #     batch=batch_idx,
        #     size=len(testloader),
        #     data=data_time.avg,
        #     bt=batch_time.avg,
        #     loss=losses.avg
        # ))
    return (losses.avg,0)

def save_checkpoint(state,is_best,checkpoint='checkpoint',filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint,filename)
    torch.save(state,filepath)
    if is_best:
        shutil.copyfile(filepath,os.path.join("/content/drive/My Drive/assess_boneage/assess_boneage/0201",'model_best.pth.tar'))

def adjust_learning_rate(optimizer,epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

if __name__ == '__main__':
    main()
