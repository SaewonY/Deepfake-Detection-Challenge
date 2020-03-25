import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from credential import token
from slacker import Slacker
from datetime import datetime
from sklearn.metrics import accuracy_score


def train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device):
    
    model.train()
    train_loss = 0.
    optimizer.zero_grad()
        
    if args.model_type == 'cnn':
        for batch_idx, ((real_img, real_label), (fake_img, fake_label)) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
            if device:
                images = torch.cat((real_img, fake_img)).to(device, dtype=torch.float)
                labels = torch.cat((real_label, fake_label)).reshape(-1, 1).to(device)
        
            # label smoothing
            # labels = torch.clamp(labels, min=0., max=0.999)

            preds = model(images)
            loss = criterion(preds, labels)
            # print(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
            # scheduler update
            if args.scheduler == 'Cosine':
                scheduler.step()


    elif args.model_type == 'lrcn':
        for batch_idx, ((real_img, real_label), (fake_img, fake_label)) in tqdm(enumerate(train_loader), total=len(train_loader)):

            if device:
                images = torch.cat((real_img, fake_img)).to(device)
                labels = torch.cat((real_label, fake_label)).reshape(-1, 1).to(device)

            preds = model(images)            
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            # print(loss.item())

            # scheduler update
            if args.scheduler == 'Cosine':
                scheduler.step()
    
    epoch_train_loss = train_loss / (len(train_loader)*2)
        
#         if (batch_idx+1) % (len(train_loader)//3) == 0:
#             val_loss, clipped_loss, clipped_loss1, val_acc = validation(model, criterion, valid_loader)

#             if val_loss < best_valid_loss:
#                 best_valid_loss = val_loss
#                 torch.save(model.state_dict(), os.path.join(save_path, 'resnet_best.pt'))
                    
#             print("batch [{}/{}]  train_loss: {:.4f}  val_loss: {:.4f}  clipped_loss: {:.4f}  clipped_loss1: {:.4f}  val_acc: {:.4f}".format(
#                     batch_idx, len(train_loader), train_loss, val_loss, clipped_loss, clipped_loss1, val_acc))
        
    return epoch_train_loss


def validation(args, model, criterion, valid_loader, device):
    
    model.eval()
    val_loss = 0.
    valid_preds = []
    valid_targets = []
    
    with torch.no_grad():
        if args.model_type == 'cnn':
            for batch_idx, ((real_img, real_label), (fake_img, fake_label)) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                            
                if device:
                    images = torch.cat((real_img, fake_img)).to(device, dtype=torch.float)
                    labels = torch.cat((real_label, fake_label)).reshape(-1, 1).to(device)
                valid_targets.append(labels.detach().cpu().numpy())
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                # print(loss.item())
                valid_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
                
                val_loss += loss.item()

        elif args.model_type == 'lrcn':
            for batch_idx, ((real_img, real_label), (fake_img, fake_label)) in tqdm(enumerate(valid_loader), total=len(valid_loader)):

                if device:
                    images = torch.cat((real_img, fake_img)).to(device)
                    labels = torch.cat((real_label, fake_label)).reshape(-1, 1).to(device)


                valid_targets.append(labels.detach().cpu().numpy())
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                # print(loss.item())
                valid_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
                
                val_loss += loss.item()
            
    epoch_val_loss = val_loss / (len(valid_loader)*2)

    valid_preds = np.concatenate(valid_preds)
    valid_targets = np.concatenate(valid_targets)
    val_acc = accuracy_score(valid_targets, np.where(valid_preds >= 0.5, 1, 0))
    val_mean = np.mean(valid_preds)

    return epoch_val_loss, val_acc, val_mean


def train_model(args, trn_cfg):
    
    model = trn_cfg['model']
    criterion = trn_cfg['criterion']
    train_loader = trn_cfg['train_loader']
    valid_loader = trn_cfg['valid_loader']
    valid_loader1 = trn_cfg['valid_loader1']
    valid_loader2 = trn_cfg['valid_loader2']
    optimizer = trn_cfg['optimizer']
    scheduler = trn_cfg['scheduler']
    device = trn_cfg['device']
    best_valid_loss = 1.0

    for epoch in range(args.n_epochs):

        print(f"epoch: {epoch+1} training starts")
        start_time = time.time()

        if epoch == 1 and args.unfreeze:
            print("model unfrozen")
            for param in model.parameters():
                param.requires_grad = True 
    
        train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device)
        
        total_val_loss = []
        total_val_acc = []
        valid_loader_list = [valid_loader, valid_loader1, valid_loader2]
        
        for i, valid_loader in enumerate(valid_loader_list):
            val_loss, val_acc, val_mean = validation(args, model, criterion, valid_loader, device)
            print("{}: {:.4f}".format(i+1, val_loss))
            total_val_loss.append(val_loss)
            total_val_acc.append(val_acc)
            
        val_loss = np.mean(total_val_loss)
        val_acc = np.mean(total_val_acc)
    
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            save_path = trn_cfg['save_path']
            torch.save(model.state_dict(), save_path)
            print("model saved as {}".format(save_path.split('/')[-1]))
    
        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.4f}  val_loss: {:.4f}  val_acc: {:.4f}  val_mean: {:.4f}  lr: {:.5f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_acc, val_mean, lr[0], elapsed))
        
        # slack notice
        slack = Slacker(token)
        slack.chat.post_message('#deepfake-train', 'Epoch {}- train_loss: {:.4f}  val_loss: {:.4f}  val_acc: {:.4f}  val_mean: {:.4f} time {}'.format(
                                                    epoch+1, train_loss, val_loss, val_acc, val_mean, datetime.now().replace(second=0, microsecond=0)
                                ))

        # scheduler update
        if args.scheduler in ['Steplr', 'Lambda']:
            scheduler.step()
        
        print()