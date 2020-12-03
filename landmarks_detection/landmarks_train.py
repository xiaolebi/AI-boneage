from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from models import *
from HandLandmarksDataset import *
from utils import AverageMeter,normalizedME,mkdir_p

parser = argparse.ArgumentParser(description='PyTorch hand landmark training')
parser.add_argument('--dataset',default='HandLandmarksDataset')
parser.add_argument('--workers',default=8,type=int,metavar='N',help='number of data loading workers (default:4)')
parser.add_argument('--epochs',default=500,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch',default=7,type=int,metavar='N',help='train batch size')
parser.add_argument('--test_batch',default=3,type=int,metavar='N',help='test batch size')
parser.add_argument('--lr','--learning-rate',default=0.001,type=float,metavar='LR',help='initial learning rate')
parser.add_argument('--drop','--dropout',default=0,type=float,metavar='Dropout',help='Dropout ratio')
parser.add_argument('--schedule',type=int,nargs='+',default=[60,100],help='Decrease learning rate at these epochs')
parser.add_argument('--gamma',type=float,default=0.1,help='LR is multiplied by gamma on schedule')
parser.add_argument('--momentum',default=0.9,type=float,metavar='M',help='momentum')
parser.add_argument('--weight_decay','--wd',default=5e-4,type=float,metavar='W',help='weight decay (default: 1e-4)')
parser.add_argument('--panelty','--pl',default=1e-3,type=float)
parser.add_argument('--checkpoint',default='/content/checkpoints',type=str,metavar='PATH',help='path to save checkpoint(default:checkpoint)')
parser.add_argument('--resume',default='',type=str,metavar='PATH',help='path to latest checkpoint(default:None)')
parser.add_argument('--depth',type=int,default=104,help='Model depth')
parser.add_argument('--cardinality',type=int,default=8,help='Model cardinality(group)')
parser.add_argument('--widen_factor',type=int,default=4,help='Widen factor 4 -> 64,8 -> 128')
parser.add_argument('--growthRate',type=int,default=12,help='Growth rate for DenseNet')
parser.add_argument('--compressionRate',type=int,default=2,help='Compression Rate(theta) for DenseNet')
parser.add_argument('--manualSeed',type=int,help='manual seed')
parser.add_argument('--e','--evaluate',dest='evaluate',action='store_true',help='evaluate model on validation set')
parser.add_argument('--gpu_id',default='0',type=str,help='id for CUDA_VISIBLE_DEVICES')

torch.cuda.empty_cache()
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

def main():
    global best_acc
    start_epoch = args.start_epoch
    if not os.path.exists(args.checkpoint):
        mkdir_p(args.checkpoint)
    print("==> Preparing dataset %s"%args.dataset)
    transform_train = transforms.Compose([
        Rescale((520,520)),
        RandomCrop((512,512)),
        RandomFlip(),
        RotateRandom(),
        # RandomBrightness(),
        ToTensor(512),
        Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    transform_test = transforms.Compose([
        Rescale((512, 512)),
        ToTensor(512),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainset = HandLandmarksDataset(csv_file='/content/dataset/retrain.csv',transform=transform_train,root_dir='/content/dataset/train')
    trainloader = data.DataLoader(trainset,batch_size=args.train_batch,shuffle=True,num_workers=args.workers)
    testset = HandLandmarksDataset(csv_file='/content/dataset/test.csv',transform=transform_test,root_dir='/content/dataset/test')
    testloader = data.DataLoader(testset,batch_size=args.test_batch,shuffle=True,num_workers=args.workers)
    model = Res152(6)
    cudnn.benchmark = True
    print('   Total params: %.2fM'%(sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.MSELoss().cuda()
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

    title = 'handlandmark_res152'
    if args.resume:
        print('==> Resume from checkpoint...')
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss,test_acc = test(testloader,model,criterion,start_epoch,use_cuda)
        print(' Test Loss: %.8f, Test Acc: %.2f'%(test_loss,test_acc))
        return

    for epoch in range(start_epoch,args.epochs):
        adjust_learning_rate(optimizer,epoch)
        print('\nEpoch: [%d | %d] LR: %f'%(epoch+1,args.epochs,state['lr']))
        train_loss = train(trainloader,model,criterion,optimizer,epoch,use_cuda)
        test_loss = test(testloader,model,criterion,start_epoch,use_cuda)
        print('train_loss:%s   test_loss:%s'%(train_loss,test_loss))
        is_best = test_loss<best_acc
        best_acc = min(test_loss,best_acc)
        if epoch%10 == 0:
            save_checkpoint({
                'epoch':epoch+1,
                'state_dict':model.state_dict(),
                'best_acc':best_acc,
                'optimizer':optimizer.state_dict(),
            },is_best,checkpoint=args.checkpoint,filename=title+'_'+str(epoch)+'.pth.tar')
    print('Best acc')
    print(best_acc)

def train(trainloader,model,criterion,optimizer,epoch,use_cuda):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for batch_idx,batch_data in enumerate(trainloader):
        data_time.update(time.time()-end)
        inputs = batch_data['image']
        targets = batch_data['landmarks']
        if use_cuda:
            inputs,targets = inputs.cuda(),targets.cuda()
        inputs,targets = torch.autograd.Variable(inputs),torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs,targets[:,:,0])
        '''
        l2_reg = torch.autograd.Variable(torch.cuda.FloatTensor(1),requires_grad=True)
        for w model.parameters():
            l2_reg = l2_reg + torch.pow(w,2).sum()
        '''
        l2_reg = 0
        loss = loss + args.panelty*l2_reg
        losses.update(loss.item(),inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()
        print('Train:({batch}/{size}) Data:{data:.3f}s | Batch:{bt:.3f}s | Loss:{loss:.4f}'.format(
            batch = batch_idx,
            size = len(trainloader),
            data = data_time.avg,
            bt = batch_time.avg,
            loss = losses.avg,
        ))
    return losses.avg

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
        if use_cuda:
            inputs,targets = inputs.cuda(),targets.cuda()
        inputs,targets = torch.autograd.Variable(inputs),torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs,targets[:,:,0])
        losses.update(loss.item(),inputs.size(0))
        batch_time.update(time.time()-end)
        end = time.time()
        print('Train:({batch}/{size}) Data:{data:.3f}s | Batch:{bt:.3f}s | Loss:{loss:.4f}'.format(
            batch=batch_idx,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg
        ))
    return losses.avg

def save_checkpoint(state,is_best,checkpoint='checkpoint',filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint,filename)
    torch.save(state,filepath)
    if is_best:
        shutil.copyfile(filepath,os.path.join(checkpoint,'model_best.pth.tar'))

def adjust_learning_rate(optimizer,epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
