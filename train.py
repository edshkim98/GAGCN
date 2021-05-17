import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
import sys
import scipy.spatial.distance
import math
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import open3d as o3d
import time
from torch_geometric.nn import knn_interpolate, fps
import json
from collections import Counter, OrderedDict
from torch.optim import AdamW
from loss import *
from utils import *
from model import *

batch_size = 2
batch_size_test = 2

def train_all(model,disc,train_loader,valid_loader, n_cls,count=None, shape = None, epochs=500, smoothing= False, save=True):
    best = 0
    patience = 100
    p = 0
    if smoothing:
        sig = 0.0
    else:
        sig = 0.0
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    
    params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b])    
    optimizer = torch.optim.Adam(params, lr=lr)
    optimizerD = torch.optim.Adam(disc.parameters(), lr=lr2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step*len(train_loader), eta_min=1e-4)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, step*len(train_loader), eta_min=1e-4)
    
    for epoch in range(epochs):
        e = epoch
        model.train()
        running_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data['points'].to(device), data['labels'].to(device)
            #UPDATE DISCRIMINATOR
            optimizerD.zero_grad()

            labels = labels.view(batch_size, -1)

            labels2 = one_hot(labels,n_cls).to(device)
            noise = torch.tensor(np.random.normal(0, sig, labels2.shape)).to(device)
            labels2 = labels2+noise

            inputs2 = torch.cat((inputs, labels2.type(torch.float64).reshape(batch_size,2048,n_cls)),axis=2)
            inputs2 = inputs2.transpose(1,2)
            real_image = inputs2
            output,em1 = disc(inputs2.float())
            ls_real = disc_loss(output,num=1, smoothing = smoothing)
            ls_real.backward(retain_graph=True)

            #Fake image
            with torch.no_grad():
                output = model(inputs.transpose(1,2))
            output_pred = output
            _,pred = torch.max(output.data,1)
            pred = pred.view(batch_size,-1)
            pred = one_hot(pred,n_cls)
            noise = torch.tensor(np.random.normal(0, sig, pred.shape))#.to(device)
            pred += noise
            pred = pred.to(device)
            inputs2 = torch.cat((inputs, pred.type(torch.float64).reshape(batch_size,2048,n_cls)),axis=2)
            inputs2 = inputs2.transpose(1,2)
            output,em2 = disc(inputs2.float())
            ls_fake = disc_loss(output, num=0, smoothing = smoothing)
            ls_fake.backward()

            ls_D = (ls_real + ls_fake)/2
            optimizerD.step()
            schedulerD.step()
            
            #UPDATE GENERATOR
            #Real image
            optimizer.zero_grad()
            #real
            inputs2 = torch.cat((inputs, labels2.type(torch.float64).reshape(batch_size,2048,n_cls)),axis=2)
            inputs2 = inputs2.transpose(1,2)
            real_image = inputs2
            #fake
            output = model(inputs.transpose(1,2))
            output_pred = output
            output2 = output.clone()
            _,pred = torch.max(output.data,1)
            pred = pred.view(batch_size,-1)
            pred = one_hot(pred,n_cls)
            noise = torch.tensor(np.random.normal(0, sig, pred.shape))#.to(device)
            pred += noise
            pred = pred.to(device)
            inputs2 = torch.cat((inputs, pred.type(torch.float64).reshape(batch_size,2048,n_cls)),axis=2)
            inputs2 = inputs2.transpose(1,2)
            
            _,em1 = disc(real_image.float())
            _,em2 = disc(inputs2.float())
            
            output_pred = [output_pred,em2]
            labels = [labels,em1]
            total_ls, emb_ls = multitask_loss(output_pred,labels,n_cls,[log_var_a,log_var_b])
            total_ls.backward()
            
            #if i%30 == 0:
            #    print("Embedding Loss: {}".format(emb_ls))
            optimizer.step()
            scheduler.step()

        std_1 = torch.exp(-log_var_a)**0.5
        std_2 = torch.exp(-log_var_b)**0.5
        
        model.eval()
        total = 0
        correct =0
        total2 = 0
        correct2 =0
        loss =[]
        loss2 = []
        valid_iou = []
        with torch.no_grad():  
            for i, data in enumerate(valid_loader):
                inputs, labels = data['points'].to(device), data['labels'].to(device)
                
                #Discriminator
                #Real
                labels = labels.view(batch_size, -1)
            
                labels2 = one_hot(labels,n_cls).to(device)
                noise = torch.tensor(np.random.normal(0, sig, labels2.shape)).to(device)
                labels2 = labels2+noise
                
                inputs2 = torch.cat((inputs, labels2.type(torch.float64).reshape(batch_size,2048,n_cls)),axis=2)
                inputs2 = inputs2.transpose(1,2)
                real_image = inputs2
                output,em1 = disc(inputs2.float())
                _,pred2 = torch.max(output.data,1)
                ls_real = disc_loss(output,num=1)
                total2 += batch_size_test
                correct2 += (pred2 == 1).sum().item()
                
                #Fake
                with torch.no_grad():
                    outputs = model(inputs.transpose(1,2))
                output_pred = outputs
                _,pred = torch.max(outputs.data,1)
                pred = pred.view(batch_size,-1)
                pred = one_hot(pred,n_cls)
                noise = torch.tensor(np.random.normal(0, sig, pred.shape))#.to(device)
                pred += noise
                pred = pred.to(device)
                inputs2 = torch.cat((inputs, pred.type(torch.float64).reshape(batch_size,2048,n_cls)),axis=2)
                inputs2 = inputs2.transpose(1,2)
                output,em2 = disc(inputs2.float())
                _,pred2 = torch.max(output.data,1)
                ls_fake = disc_loss(output, num=0)
                
                ls_total = (ls_real + ls_fake)/2
                loss2.append(ls_total.item())
                total2 += batch_size_test 
                correct2 += (pred2 == 0).sum().item()
                
                #Generator
                outputs = model(inputs.transpose(1,2))
                ls = gagcn_loss(outputs,labels,n_cls, smoothing = True)

                loss.append(ls.item())
                _,pred = torch.max(outputs.data,1) #why? because the first return is for the values(probabilities) 
                total += labels.size(0) * labels.size(1) 
                correct += (pred == labels).sum().item()
                pred = pred.cpu().detach().numpy()
                ious = calculate_shape_IoU(pred,labels.cpu().detach().numpy(),n_cls)
                valid_iou.append(np.mean(ious))
            
            #Generaotr    
            acc = 100. * correct / total
            loss_final = np.mean(loss)
            iou_final = np.mean(valid_iou)
            print("Epoch: {}".format(e))
            print("Generator: The validation accuracy: {:.6f} and the validation loss: {:.6f} and the validation iou: {:.6f}".format(acc,loss_final,iou_final))
            print("std1 {} std2 {}".format(std_1,std_2))
            #Discriminator
            acc = 100. * correct2 / total2
            loss_final = np.mean(loss2)
            print("Discriminator: The validation accuracy: {:.6f} and the validation loss: {:.6f}".format(acc,loss_final))
            
            if best < iou_final:
                best = iou_final
                p = 0
                if save:
                    torch.save(model.state_dict(), "/home/edshkim98/shkim/pretrained_group_final/partseg_"+str(shape)+'_'+str(count)+".pth")
                else:
                    pass
            else:
                p +=1
                print("Patience: {} / {}".format(p,patience))
                if p == patience:
                    print("Stopped due to convergence")
                    return best
            print("Valid best iou: ",best)
    return best

if __name__ == "__main__":
    dataset = {"02691156": "Airplane", "02773838": "Bag", "02954340": "Cap", "02958343": "Car", "03001627": "Chair",
           "03261776": "Earphone", "03467517": "Guitar", "03624134": "Knife", "03636649": "Lamp", "03642806": "Laptop",
           "03790512": "Motorbike", "03797390": "Mug", "03948459": "Pistol", "04099429": "Rocket", "04225987": "Skateboard",
           "04379243": "Table"}
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    print("Training on ShapeNet dataset")
    print("#############################")
    total_final = 0
    total_num = 0 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for o in range(15,len(dataset.keys())):
        shape = list(dataset.keys())[-3]
        path = r'/home/edshkim98/shkim/shape_data/'+shape+'/'#+shape
        new_path =path+'/points_label'
        n_cls = seg_num[-3]

        valid_files = '/home/edshkim98/shkim/shape_data/train_test_split/shuffled_val_file_list.json'
        train_files = '/home/edshkim98/shkim/shape_data/train_test_split/shuffled_train_file_list.json'
        test_files = '/home/edshkim98/shkim/shape_data/train_test_split/shuffled_test_file_list.json'

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        train_files = read_json(train_files)
        val_files = read_json(valid_files)
        test_files = read_json(test_files)

        train_xs = []
        val = []
        test = []
        for i in train_files:
            if shape in i:
                train_xs.append(i)
        for i in val_files:
            if shape in i:
                val.append(i)
        for i in test_files:
            if shape in i:
                test.append(i)
        total = train_xs+val+test
        np.random.shuffle(total)
        #Train
        train_xs = train_xs+val#np.array(total)[idx]
        #Val&Test
        val = test
        train_dataset = CustomDataset(path,train_xs, transform =train_transforms,valid = False)
        val_dataset = CustomDataset(path, val,transform=None, valid=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True, num_workers =2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, drop_last = True, num_workers =2)


        print('######### Dataset class created #########')
        print("Shape: {} Batch size: {}".format(shape,batch_size))
        print('Number of images: ', len(train_xs)+len(val))
        print('Sample image shape: ', train_dataset[0]['points'].shape)
        print('train size in no of batch: ',len(train_loader))
        print("test size in no of batch: ",len(val_loader))
        print('train size: ',len(train_loader)*batch_size)
        print("valid size: ",len(val_loader)*batch_size_test)

            #########################

            ###########################
        best = 0
        total = len(val_loader)*batch_size_test

        if len(train_loader)*batch_size > 1500:
            cnts = 1
        else:
            cnts=2

        for cnt in range(cnts):
            smoothing=False
            bias = True
            dilation_rate = [1,3,6,6,6,1]
            pyramid = [False,False,False,False,False]
            dilation = 'order'
            lr = 0.005
            lr2 = 0.01
            step = 5

            se = False
            gagcn = GAGCN(n_cls,k=16,pyramid=pyramid,dilation='order', dilation_rate = dilation_rate, norm='group', bias= bias)
            torch.cuda.set_device(0)
            gagcn.to(device)
            pytorch_total_params = sum(p.numel() for p in gagcn.parameters())
            print("Number of parameters (segmentation): ", pytorch_total_params)

            se = False
            disc = Discriminator(k=12, dilation='order', dilation_rate= [1,3,3,3], n_cls= n_cls, out_dims= [128,256,256,512],pyramid=[False,False,False,False],norm='group')
            disc.to(device)
            pytorch_total_params = sum(p.numel() for p in disc.parameters())
            print("Number of parameters (discriminator): ", pytorch_total_params)
            print("Disc smoothing: ",smoothing)

            best_curr = train_all(gagcn,disc,train_loader, val_loader, n_cls, count=cnt,shape=shape, smoothing = smoothing,save=False)
            if best_curr > best:
                best = best_curr
                print("###############################################")
                print("Best Setting-> Shape {} IoU {} count {}".format(shape, best, cnt))
                print("Best: {}%".format(best*100))
                print("###############################################")