import time
import os
import argparse
import shutil
import torch
import torch.optim as optim
from dataset import *
from models.model import *
from config_parser import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from FocalLoss2d import FocalLoss2d

parser = argparse.ArgumentParser(description='BoneAge')
parser.add_argument('--batch_size',type=int,default=6,help='input batch size for training')
parser.add_argument('--lr',type=float,default=1e-5,help='initial learning rate')
parser.add_argument('--test_batch_size',type=int,default=12,metavar='N',help='input batch size for testing')
parser.add_argument('--start_epoch',type=int,default=0,help='start epoch')
parser.add_argument('--epochs',type=int,default=50,metavar='N',help='number of epochs to train')
parser.add_argument('--seed',type=int,default=212,metavar='S',help='random seed')
parser.add_argument('--log_interval',type=int,default=10,metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--resume',type=str,default='/content/checkpoints/resume/model_best.pth_full_hand.tar',help='resume training')

args = parser.parse_args()
state = {k:v for k,v in args._get_kwargs()}
args.cuda = torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(epoch,model,optimizer,train_loader,iters):
    model.train()
    criterion = FocalLoss2d(torch.FloatTensor(class_weight),alpha=2).cuda()
    dice_co = 0
    count = 0

    for batch_idx,(data,target) in enumerate(train_loader):
        data = Variable(data.cuda())
        target = Variable(target.cuda())
#         data = Variable(data)
#         target = Variable(target)
        output = model(data)
        optimizer.zero_grad()
        _,pred = torch.max(output,1)
        dice_coef = compute_dice(pred,target)
        dice_co += dice_coef
        loss = criterion(output,target[:,:,:,0]//255)+Variable(torch.FloatTensor([10.0-10.0*dice_coef]).cuda())
#         loss = criterion(output, target[:, :, :, 0] // 255) + Variable(torch.FloatTensor([10.0 - 10.0 * dice_coef]))
        loss.backward()
        optimizer.step()
        count += torch.sum(pred.data[:,:,:] == (target.data[:,:,:,0]//255))
        if batch_idx % args.log_interval == 0 and not batch_idx == 0:
            print("Train  batch [{}/{} ({:.0f}%)]  Loss:{:.4f}  acc:{:.2f}%  ave dice coef:{:.4f}".format(
                # epoch,args.epochs,
                batch_idx*len(data),len(train_loader.dataset),
                100.0*batch_idx / len(train_loader),loss.data[0],
                100.0*count / args.log_interval / torch.numel(target.data[:,:,:,0]),
                dice_co/args.log_interval
            ))
            iters += 1
            dice_co = 0
            count = 0
    return loss.data[0],iters


def compute_dice(pred,target):
    dice_count = torch.sum(pred.data[:,:,:].type(torch.ByteTensor)&target.data[:,:,:,0].type(torch.ByteTensor)//255)
    dice_sum = (1.0/255*torch.sum(target.data[:,:,:,0].type(torch.ByteTensor))+1.0*torch.sum(pred.data[:,:,:].type(torch.ByteTensor)))
    return (2*dice_count+1.0)/(dice_sum+1.0)

def save_checkpoint(state,is_best,epoch,iters):
    filename = '/content/checkpoints/checkpoint_' + str(epoch) + '_' + str(iters) +'.pth.tar'
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'/content/checkpoints/model_best.pth.tar')

def resume(ckpt,model):
    if os.path.isfile(ckpt):
        print('==> loading checkpoint {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
#         args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']
        iters = checkpoint['iters']
        print("==> loaded checkpoint '{}'".format(args.resume))
        return model,optimizer,args.start_epoch,best_loss,iters
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

def adjust_lr(optimizer,epoch,decay=5):
#     lr = args.lr*(0.1**(epoch//decay))
    if epoch%10 == 0 and epoch != 0:
        state['lr'] *= 0.5
        for param in optimizer.param_groups:
            param['lr'] = state['lr']
    return state['lr']

def main():
    cfp = config_parser()
    root = cfp.get("PATH","root")
    train_filename = cfp.get("PATH","train")
    mask_filename = cfp.get("PATH","mask")

    kwargs = {'num_workers':1,'pin_memory':True} if args.cuda else {}
    train_image_list = os.listdir(os.path.join(root,mask_filename))
    train_filename_list,eval_filename_list = augement_train_valid_split(train_image_list,0.1,shuffle=True)
    HandSet = HandDataSet(root,train_filename,mask_filename,transform=True,trainable=True,train_image_list=train_filename_list)
    train_loader = DataLoader(HandSet,shuffle=True,batch_size=args.batch_size,**kwargs)
    model = UNet(num_class)
    print(' Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999))
    best_loss = 1e+5
    iters = 0

    if args.resume:
        model,optimizer,args.start_epoch,best_loss,iters = resume(args.resume,model)
        args.start_epoch = 0
        args.lr = 1e-5

    for epoch in range(args.start_epoch,args.epochs):
        lr = adjust_lr(optimizer,epoch,decay=5)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch+1,args.epochs,lr))
        loss,iters = train(epoch,
                           model,
                           optimizer,
                           train_loader,
                           iters)
        is_best = loss<best_loss
        best_loss = min(best_loss,loss)
        state = {
            'epoch:':epoch,
            'state_dict':model.state_dict(),
            'optimizer':optimizer,
            'loss':best_loss,
            'iters':iters,
        }
        save_checkpoint(state,is_best,epoch,iters)

if __name__ == '__main__':
    main()
