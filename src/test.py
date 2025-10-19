import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
import csv
#from utils.sbi_test import SBI_Dataset
from utils.data import Dataset
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def main(args):
    cfg=load_json(args.config)

    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    print(device)
    batch_size=cfg['batch_size']
    val=cfg['subdatasets_name']
    
    model=Detector()
    
    model=model.to(device)
    load_file=''
    K_state_dict = torch.load(load_file)["model"]
    model.net_all.load_state_dict(K_state_dict)
   
    model.eval()
    

    results_dir=load_file.split('/weights/')[0]+'/results'
    mkdir(results_dir)
    model.train(mode=False)
    model_name='DeCoF'

   
    
    print("Testing on {}".format(val))
    val_dataset=Dataset(phase='test',data_name=val)


    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=20,
                        pin_memory=True,
                        drop_last=False
                        )

    output_dict=[]
    target_dict=[]
    output_dict_acc=[]
    np.random.seed(seed)
    for step,data in enumerate(tqdm(val_loader)):
        img=data['img'].to(device, non_blocking=True).float()
        target=data['label'].to(device, non_blocking=True).long()

        with torch.no_grad():

            output=model(img)

        output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
        output_dict_acc+=(F.log_softmax(output,dim=1).argmax(dim=1).cpu().data.numpy().tolist())
        target_dict+=target.cpu().data.numpy().tolist()

    y_true, y_pred,y_pred_acc = np.array(target_dict), np.array(output_dict), np.array(output_dict_acc)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred_acc[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], y_pred_acc[y_true == 1])
    acc = accuracy_score(y_true,y_pred_acc)
    ap = average_precision_score(y_true, y_pred)
    auc=roc_auc_score(y_true, y_pred)



    print("({}) acc: {}; ap: {};  r_acc: {}, f_acc: {}, auc:{}".format(val, acc, ap, r_acc, f_acc,auc))


        
if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n',dest='session_name')
    args=parser.parse_args()
    main(args)
            
