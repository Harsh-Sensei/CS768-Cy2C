import os.path
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid,TUDataset
from torch_geometric.nn import GATConv, GCNConv,GINConv
import numpy as np
import igraph as ig
from torch_geometric.nn import global_mean_pool,global_add_pool
from functools import reduce
from sklearn.model_selection import KFold,StratifiedKFold
import torch.optim as optim
import csv
import pandas as pd
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator


class Trainer(object):    
    def __init__(self, model_name, dataset_name,dataset,device,MODEL_CLASS,num_node_features,num_classes, train_loader, valid_loader, test_loader,batch_size=128,hidden_dim=138,lr=7e-4,n_layer=3,num_workers=4,opt_fun=1,eval_metric=None,dataset_name2=None,dropout=0.0,decay=0.0, n_fold=1,drop_mid=0.0):
        super(Trainer, self).__init__()        
        self.num_workers=num_workers
        self.lr=lr
        self.opt_fun=opt_fun
        self.n_layer=n_layer
        self.hidden_dim=hidden_dim
        self.model_name = model_name
        self.batch_size=batch_size
        self.dataset_name = dataset_name
        self.num_node_features=num_node_features
        self.num_classes=num_classes
        self.MODEL_CLASS=MODEL_CLASS                      
        self.device = device        
        self.dataset = dataset
        self.check_path=f'./dataset/{dataset_name}/checkpoint'
        self.result_path =f'./dataset/{dataset_name}/result' 
        self.modelsave_path=f'{self.check_path}/{self.model_name}'
        self.dataset_name2=dataset_name2
        self.eval_metric=eval_metric
        self.dropout=dropout
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.valid_loader=valid_loader
        self.n_fold=n_fold
        self.drop_mid=drop_mid
         #전체에서 1번만
        self.decay=decay
        if os.path.isdir(self.check_path):
            print('checkpoint')
        else:
            os.makedirs(self.check_path)
            
        if os.path.isdir(self.result_path):
            print('result')
        else:
            os.makedirs(self.result_path)
            
        if os.path.isdir(self.modelsave_path):
            print('model_name')
        else:
            os.makedirs(self.modelsave_path) 
    
    def load_model(self):
        #model = GCN_3_H1_CON(hidden_channels=128).to(self.device)      
        if self.drop_mid>0:
            model=self.MODEL_CLASS(self.hidden_dim,self.num_node_features,self.num_classes,n_layer=self.n_layer,dropout_ratio=self.dropout,drop_mid=self.drop_mid).to(self.device) 
        else:
            model=self.MODEL_CLASS(self.hidden_dim,self.num_node_features,self.num_classes,n_layer=self.n_layer,dropout_ratio=self.dropout).to(self.device)  
        return model
    
    # 요게 이제 train & save 까지의 메인 함수.
    def train(self):                              
        self.total_results = {
                'mainfold_index': [],
                'subfold_index': [],
                'best_epoch': [],
                'best_val_loss' :[],
                'best_val_acc' :[],
                'final_epoch': [],
                'final_val_loss' :[],
                'final_val_acc' :[],
                'best_test_loss' :[],
                'best_test_acc' :[],
                'final_test_loss' :[],
                'final_test_acc' :[]
            }       
        for mainfold_idx in range(self.n_fold):
                       
                              
            for subfold_idx in [0]:
                self.epoch_results = {
                    'val_loss': [],
                    'val_acc': [],
                    'train_loss': [],
                    'epoch': [],
                    'best_epoch': [],
                    'best_val_loss' :[],
                    'best_val_acc' :[]
                }
                 
                
                #1. initialize model , optimizer, loss function, shceduler
                self.model = self.load_model()
                def initialize_weights(m):
                    if hasattr(m, 'weight') and m.weight.dim() > 1:
                        nn.init.xavier_uniform_(m.weight.data)

                        self.model.apply(initialize_weights)
                minimum_lr=1E-6
                if self.opt_fun ==1:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=self.decay)
                elif self.opt_fun==2:
                    self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr)
                elif self.opt_fun==3:
                    import torch_optimizer as Optim
                    self.optimizer =Optim.AdamP(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',factor=0.8, patience=25, min_lr=minimum_lr)
                self.criterion = torch.nn.BCEWithLogitsLoss()
                self.evaluator=Evaluator(self.dataset_name2)
                

                #print(f'The model has {count_parameters(model):,} trainable parameters')
                best_valid_loss = 10000
                best_valid_acc = 0
                best_epoch = 0
                patience = 0
                valid_curve=[]
                test_curve=[]
                for epoch in range(0,50000):
                    train_loss=self.train_train()
                    valid_acc = self.test(self.valid_loader)
                    #test_acc = self.test(self.test_loader)
                    valid_acc=valid_acc[self.eval_metric]
                    #test_acc=test_acc[self.eval_metric]
                    self.scheduler.step(valid_acc)

                    
                    
                    if best_valid_acc < valid_acc :
                        best_valid_acc = valid_acc
                        best_epoch = epoch                        
                        torch.save(self.model.state_dict(),f'{self.check_path}/{self.model_name}/{mainfold_idx}_{subfold_idx}_best_model.pt')
                        patience=0
                    else:
                        patience+=1

                        
                    if self.optimizer.param_groups[0]['lr'] <= minimum_lr or patience>100:                        
                        print(f'Mainfold_index: {mainfold_idx}, Subfold_index:{subfold_idx}')     
                        test_acc_final = self.test(self.test_loader)   
                        test_acc_final = test_acc_final[self.eval_metric]
                        torch.save(self.model.state_dict(),f'{self.check_path}/{self.model_name}/{mainfold_idx}_{subfold_idx}_final_model.pt')
                        self.model.load_state_dict(torch.load(f'{self.check_path}/{self.model_name}/{mainfold_idx}_{subfold_idx}_best_model.pt'))                        
                        test_acc = self.test(self.test_loader)
                        test_acc=test_acc[self.eval_metric]
                        #results=pd.DataFrame(self.epoch_results)
                        #results.to_csv(f'{self.check_path}/{self.model_name}_{mainfold_idx}_{subfold_idx}_epoch_results.csv', na_rep='NaN')
                        # results2=pd.DataFrame(self.overall_results)
                        # results2.to_csv(f'{self.check_path}/{self.model_name}_{mainfold_idx}_{subfold_idx}_overall_results.csv', na_rep='NaN')

                        self.total_results['mainfold_index'].append(mainfold_idx)
                        self.total_results['subfold_index'].append(subfold_idx)
                        
                        self.total_results['best_epoch'].append(best_epoch)
                        self.total_results['best_val_acc'].append(best_valid_acc)
                        
                        self.total_results['final_epoch'].append(epoch)
                        self.total_results['final_val_acc'].append(valid_acc)
                        
                        self.total_results['best_test_acc'].append(test_acc)  
                        
                        self.total_results['final_test_acc'].append(test_acc_final)
                        
                        print(f'main & sub ==={mainfold_idx},{subfold_idx},best acc & loss==,{test_acc:.4f},final acc & loss=={test_acc_final:.4f},best_epoch=={best_epoch},final_epoch=={epoch}')
#                         f_txt = open(f'{self.result_path}/{self.model_name}_results.txt', 'a')
#                         f_txt.write(f'{mainfold_idx}_{subfold_idx}_test_{test_acc}_{test_loss}_finaltest_{test_acc_final}_{test_loss_final}_epoch_{best_epoch}_{epoch}.\n' )
#                         f_txt.close() 
                        break
                              
        #전체에서 1번만
#         if os.path.isdir(f'{self.check_path}/{self.model_name}'):
#             print('end')
#         else:
#             os.makedirs(f'{self.check_path}/{self.model_name}')
        #np.array(self.overall_results['val_acc']).mean(), 
        print(self.model)
        #results3=pd.DataFrame(self.total_results)
        #results3.to_csv(f'{self.check_path}/{self.model_name}_{mainfold_idx}_total_results.csv', na_rep='NaN')
        f_txt = open(f'{self.result_path}/model_comparision.txt', 'a')
        #print(np.array(self.total_results['best_test_acc']).mean())
        bm=np.array(self.total_results['best_test_acc']).mean()
        bstd=np.array(self.total_results['best_test_acc']).std()
        m=np.array(self.total_results['final_test_acc']).mean()
        std=np.array(self.total_results['final_test_acc']).std()
        f_txt.write(f'{self.model_name}_besttest_{bm}_{bstd}_finaltest_{m}_{std}\n')
        f_txt.close() 
            
#             final_mean_test_acc=np.array(self.total_results['final_test_acc']).mean()
#             final_std_test_acc=np.array(self.total_results['final_test_acc']).std()                              
#             mean_test_acc=np.array(self.total_results['best_test_acc']).mean()
#             std_test_acc=np.array(self.total_results['best_test_acc']).std()
#             mean_valid_acc=np.array(self.total_results['best_val_acc']).mean()
#             std_valid_acc=np.array(self.total_results['best_val_acc']).std()
            
#             f_txt.close()        
                              
    def test(self, loader):
        self.model.eval()
        correct = 0 
        total_loss=0
        y_true=[]
        y_pred=[]
        with torch.no_grad():
            for data in loader:  # Iterate in batches over the training/test dataset.
                #print(len(data))
                data=data.to(self.device)
                out = self.model(data.x, data.edge_index, data.cycle_index, data.batch, data.edge_attr) 
                y_true.append(data.y.view(out.shape).detach().cpu())
                y_pred.append(out.detach().cpu())
                #pred = out.argmax(dim=1)  # Use the class with highest probability.
                #correct += int((pred == data.y).sum())  # Check against ground-truth labels.
                #loss = self.criterion(out, data.y)
                #total_loss+=loss.item()
        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return self.evaluator.eval(input_dict)  # Derive ratio of correct predictions.
                              
    def train_train(self):          
        self.model.train()
        total_loss=0
        for data in self.train_loader:  # Iterate in batches over the training dataset.
            self.optimizer.zero_grad()
            #print(data.y.shape,data.y[0])
            data=data.to(self.device)            
            out = self.model(data.x, data.edge_index, data.cycle_index, data.batch, data.edge_attr)  # Perform a single forward pass.
            is_labeled = data.y == data.y
            #print(out.shape,out[0])
            loss =self.criterion(out[is_labeled], data.y.float()[is_labeled])  # Compute the loss.
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            total_loss+=loss.item()
        return total_loss/len(self.train_loader.dataset)            