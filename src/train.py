import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils.data import Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector

def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(args):
    cfg=load_json(args.config)

    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    #print(device)

    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    n_frames=cfg['n_frames']
    subdatasets_name=cfg['subdatasets_name']
    
    
    model=Detector()
    model=model.to(device)
    

    
    train_dataset=Dataset(phase='train',data_name=subdatasets_name,image_size=image_size,n_frames=n_frames)
    val_dataset=Dataset(phase='val',data_name=subdatasets_name,image_size=image_size,n_frames=n_frames)
   
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=20,
                        pin_memory=True,
                        
                        )
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=20,
                        pin_memory=True,
                        )

    iter_loss=[]
    train_losses=[]
    test_losses=[]
    train_accs=[]
    test_accs=[]
    val_accs=[]
    val_losses=[]
    n_epoch=cfg['epoch']
    lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    last_loss=99999


    now=datetime.now()
    save_path='./output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    os.mkdir(save_path)
    os.mkdir(save_path+'weights/')
    os.mkdir(save_path+'logs/')
    logger = log(path=save_path+"logs/", file="losses.logs")

    criterion=nn.CrossEntropyLoss()
    print(len(val_loader),len(train_loader))

    last_auc=0
    last_val_auc=0
    weight_dict={}
    n_weight=20
    pred_list, target_list = [], []
    for epoch in range(n_epoch):
       
       
        train_loss=0.
        train_acc=0.
        for step,data in enumerate(tqdm(train_loader)):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            output=model.training_step(img, target)
            loss=criterion(output,target)
            loss_value=loss.item()
            iter_loss.append(loss_value)
            train_loss+=loss_value

            #acc=compute_accuray(F.log_softmax(output,dim=1),target)
            pred_list+=(F.log_softmax(output,dim=1).argmax(dim=1).cpu().data.numpy().tolist())
            #print(pred_list)
            
            target_list+=target.cpu().data.numpy().tolist()
            
        lr_scheduler.step()
        train_losses.append(train_loss/len(train_loader))
        train_acc=accuracy_score(target_list,pred_list)

        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
                        epoch+1,
                        n_epoch,
                        train_loss/len(train_loader),
                        train_acc,
                        )

    
       
        if  (epoch% 1)==0:
            model.eval()
            val_loss=0.
            val_acc=0.
            output_dict=[]
            target_dict=[]
            output_dict_acc=[]
           
            for step,data in enumerate(tqdm(val_loader)):
                img=data['img'].to(device, non_blocking=True).float()
                target=data['label'].to(device, non_blocking=True).long()
                
                with torch.no_grad():

                    output=model(img)
                
                
                output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
                output_dict_acc+=(F.log_softmax(output,dim=1).argmax(dim=1).cpu().data.numpy().tolist())
                target_dict+=target.cpu().data.numpy().tolist()
            
            #eval_accs.append(val_acc/len(val_loader))
            val_auc=roc_auc_score(target_dict,output_dict)
            val_acc=accuracy_score(target_dict, output_dict_acc)
            log_text+=" val acc: {:.4f}, val auc: {:.4f}".format(
                            
                            val_acc,
                            val_auc
                            )
        

            if len(weight_dict)<n_weight:
                save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                weight_dict[save_model_path]=val_auc
                torch.save({
                        "model":model.net_all.state_dict(),
                        "optimizer":model.optimizer.state_dict(),
                        "epoch":epoch
                    },save_model_path)
                last_val_auc=min([weight_dict[k] for k in weight_dict])

            elif val_auc>=last_val_auc:
                save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                for k in weight_dict:
                    if weight_dict[k]==last_val_auc:
                        del weight_dict[k]
                        os.remove(k)
                        weight_dict[save_model_path]=val_auc
                        break
                torch.save({
                        "model":model.net_all.state_dict(),
                        "optimizer":model.optimizer.state_dict(),
                        "epoch":epoch
                    },save_model_path)
                last_val_auc=min([weight_dict[k] for k in weight_dict])
        
        logger.info(log_text)
        
if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n',dest='session_name')
    args=parser.parse_args()
    main(args)
        
