import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
from math import log10 as log
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
from datasets import GoProDataset
import time

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 2400)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 2)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-scale","--scale",type=int, default=1)
args = parser.parse_args()

#Hyper Parameters
METHOD = "DMPHN_recon_%s_1"%args.scale
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_deblur_images(images, iteration, epoch):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_deblur.png"
    torchvision.utils.save_image(images, filename)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    print("init data folders")
    
    train_writer = SummaryWriter('runs/'+METHOD+'/train')
    test_writer = SummaryWriter('runs/'+METHOD+'/test')

    encoder_lv1 = models.Encoder()

    decoder_lv1 = models.Decoder()
    decoder_recon = models.Decoder()
    
    encoder_lv1.apply(weight_init).cuda(GPU)    

    decoder_lv1.apply(weight_init).cuda(GPU)  
    decoder_recon.apply(weight_init).cuda(GPU)  
    
    encoder_lv1_optim = torch.optim.Adam(encoder_lv1.parameters(),lr=LEARNING_RATE)
    encoder_lv1_scheduler = StepLR(encoder_lv1_optim,step_size=1000,gamma=0.1)

    decoder_lv1_optim = torch.optim.Adam(decoder_lv1.parameters(),lr=LEARNING_RATE)
    decoder_lv1_scheduler = StepLR(decoder_lv1_optim,step_size=1000,gamma=0.1)
    decoder_recon_optim = torch.optim.Adam(decoder_lv1.parameters(),lr=LEARNING_RATE)
    decoder_recon_scheduler = StepLR(decoder_lv1_optim,step_size=1000,gamma=0.1)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1_%s.pkl"%(args.start_epoch-1))):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1_%s.pkl"%(args.start_epoch-1))))
        print("load encoder_lv1 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1_%s.pkl"%(args.start_epoch-1))):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1_%s.pkl"%(args.start_epoch-1))))
        print("load decoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_recon_%s.pkl"%(args.start_epoch-1))):
        decoder_recon.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_recon_%s.pkl"%(args.start_epoch-1))))
        print("load decoder_recon success")
    
    if os.path.exists('./checkpoints/' + METHOD) == False:
        os.system('mkdir ./checkpoints/' + METHOD)    
            
    for epoch in range(args.start_epoch, EPOCHS):
        encoder_lv1_scheduler.step(epoch)

        decoder_lv1_scheduler.step(epoch)
        decoder_recon_scheduler.step(epoch)
        
        print("Training...")
        
        train_dataset = GoProDataset(
            blur_image_files = './datas/GoPro/train_blur_file.txt',
            sharp_image_files = './datas/GoPro/train_sharp_file.txt',
            root_dir = './datas/',
            crop = True,
            crop_size = IMAGE_SIZE,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
        start = 0
        
        loss_sum = 0
        loss_recon_sum = 0
        
        for iteration, images in enumerate(train_loader):            
            mse = nn.MSELoss().cuda(GPU)          
            
            gt = Variable(images['sharp_image'] - 0.5).cuda(GPU)            
            H = gt.size(2)
            W = gt.size(3)

            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)

            feature_lv1 = encoder_lv1(images_lv1)
            deblur_image = decoder_lv1(feature_lv1)
            recon_image = decoder_recon(feature_lv1)
            
#             print(deblur_image.size())
            
            loss_lv1 = mse(deblur_image, gt)
            loss_recon = mse(recon_image, images_lv1)
            
            loss = loss_lv1+args.scale*loss_recon
            
            loss_sum += loss_lv1.item()
            loss_recon_sum += loss_recon.item()
            
            encoder_lv1.zero_grad()

            decoder_lv1.zero_grad()

            loss.backward()

            encoder_lv1_optim.step()

            decoder_lv1_optim.step()
            
            if (iteration+1)%10 == 0:
                stop = time.time()
                psnr = 10*log(10/loss_sum)
                print("epoch:", epoch, "iteration:", iteration+1, "loss:%.4f"%loss.item(), "Recon loss:%.4f"%loss_recon.item(), 'time:%.4f'%(stop-start))
                print("Total loss:%.4f"%(loss_sum))
                print("Recon loss:%.4f"%(loss_recon_sum))
                print("Average psnr:%.4f"%(psnr))
                train_writer.add_scalar('Train_loss', loss.item(), (epoch)*len(train_loader)+iteration)
                train_writer.add_scalar('Train_recon_loss', loss_recon.item(), (epoch)*len(train_loader)+iteration)
                train_writer.add_scalar('Train_psnr', psnr, (epoch)*len(train_loader)+iteration)
                loss_sum = 0
                loss_recon_sum = 0
                start = time.time()
                
        if (epoch)%100==0:
            if os.path.exists('./checkpoints/' + METHOD + '/epoch' + str(epoch)) == False:
            	os.system('mkdir ./checkpoints/' + METHOD + '/epoch' + str(epoch))
            
            print("Testing...")
            test_dataset = GoProDataset(
                blur_image_files = './datas/GoPro/test_blur_file.txt',
                sharp_image_files = './datas/GoPro/test_sharp_file.txt',
                root_dir = './datas/',
                transform = transforms.Compose([
                    transforms.ToTensor()
                ]))
            test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
            loss_sum = 0
            test_time = 0.0       		
            for iteration, images in enumerate(test_dataloader):
                with torch.no_grad():         
                    gt = Variable(images['sharp_image'] - 0.5).cuda(GPU) 
                    images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
                    start = time.time()
                    H = images_lv1.size(2)
                    W = images_lv1.size(3)                    
                    
                    feature_lv1 = encoder_lv1(images_lv1)
                    deblur_image = decoder_lv1(feature_lv1)

                    stop = time.time()
                    
                    loss = mse(deblur_image, gt)
            
                    loss_sum += loss.item()
                    test_time += stop - start
#                     print('RunTime:%.4f'%(stop-start), '  Average Runtime:%.4f'%(test_time/(iteration+1)))
#                     save_deblur_images(deblur_image.data + 0.5, iteration, epoch)
            
            loss_sum_avg = loss_sum/(len(test_dataloader))
            psnr = 10*log(1/loss_sum_avg)
            print("Average psnr:%.4f"%(psnr))
            print("Average loss:%.4f"%(loss_sum_avg))
            test_writer.add_scalar('Test_loss', loss_sum_avg, (epoch+1)*len(train_loader))
            test_writer.add_scalar('Test_psnr', psnr, (epoch+1)*len(train_loader))
                    
            torch.save(encoder_lv1.state_dict(),str('./checkpoints/' + METHOD + "/encoder_lv1_%s.pkl"%epoch))

            torch.save(decoder_lv1.state_dict(),str('./checkpoints/' + METHOD + "/decoder_lv1_%s.pkl"%epoch))
            torch.save(decoder_recon.state_dict(),str('./checkpoints/' + METHOD + "/decoder_recon_%s.pkl"%epoch))
                

if __name__ == '__main__':
    main()

        

        

