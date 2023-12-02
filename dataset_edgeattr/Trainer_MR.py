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


class Trainer(object):    
    def __init__(self, model_name, dataset_name,dataset,device,MODEL_CLASS,num_node_features,num_classes,batch_size=128,hidden_dim=138,lr=7e-4,n_layer=3,num_workers=4,opt_fun=1,dropout=0,decay=0,drop_mid=0.0):
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
        self.dropout=dropout
        self.drop_mid=drop_mid
        self.decay=decay
         #전체에서 1번만
            
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
             
            
    def load_dataloader(self,dataset,train_index,valid_index,test_index):
        train_dataset=[]
        test_dataset=[]
        valid_dataset=[]
        for i in train_index:
            train_dataset.append(dataset[i])
        for i in test_index:
            test_dataset.append(dataset[i])
        for i in valid_index:
            valid_dataset.append(dataset[i])    
                              
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)                             
        return train_loader, valid_loader, test_loader
    
    def load_model(self):
        if self.drop_mid>0:
        #model = GCN_3_H1_CON(hidden_channels=128).to(self.device)                              
            model=self.MODEL_CLASS(self.hidden_dim,self.num_node_features,self.num_classes,n_layer=self.n_layer,dropout_ratio=self.dropout,drop_mid=self.drop_mid).to(self.device)  
        else:
            model=self.MODEL_CLASS(self.hidden_dim,self.num_node_features,self.num_classes,n_layer=self.n_layer,dropout_ratio=self.dropout).to(self.device)  
        return model
    
    def load_index(self,j,k):       
        print('load mainfold, subfold==', j, k)
        test_index = torch.as_tensor(np.loadtxt(f'./dataset/{self.dataset_name}/kfold_data/test_idx-{j}.txt',dtype=np.int32), dtype=torch.long)
        train_index = torch.as_tensor(np.loadtxt(f'./dataset/{self.dataset_name}/kfold_data/train_total_{j}/train_idx-{k}.txt',dtype=np.int32), dtype=torch.long)
        valid_index = torch.as_tensor(np.loadtxt(f'./dataset/{self.dataset_name}/kfold_data/train_total_{j}/valid_idx-{k}.txt',dtype=np.int32), dtype=torch.long)    
        all_index = reduce(np.union1d, (train_index, valid_index, test_index))
        assert len(self.dataset) == len(all_index)
        return train_index, valid_index, test_index
    
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
        for mainfold_idx in range(10):
                       
                              
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

#                 self.overall_results = {
#                     'val_loss': [],
#                     'val_acc': [],
#                     'test_loss': [],
#                     'test_acc': [],
#                     'train_loss':[],
#                     'train_acc': [],
#                     'epoch': []
#                 }                      
                
                #1. initialize model , optimizer, loss function, shceduler
                self.model = self.load_model()
                def initialize_weights(m):
                    if hasattr(m, 'weight') and m.weight.dim() > 1:
                        nn.init.xavier_uniform_(m.weight.data)

                        self.model.apply(initialize_weights)            
            
                minimum_lr=1E-6
                if self.opt_fun ==1:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
                elif self.opt_fun==2:
                    self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=0.8, patience=25, min_lr=minimum_lr)
                self.criterion = nn.CrossEntropyLoss()
                
                #2. Load mainfold & subfold index #=-=========
                train_index,valid_index,test_index = self.load_index(mainfold_idx,subfold_idx)
                self.train_loader, valid_loader, test_loader = self.load_dataloader(self.dataset,train_index,valid_index,test_index)
                
                #print(f'The model has {count_parameters(model):,} trainable parameters')
                best_valid_loss = 10000
                best_valid_acc = 0
                best_epoch = 0
                patience = 0
                
                for epoch in range(0,50000):
                    train_loss=self.train_train()
                    valid_acc,valid_loss = self.test(valid_loader)
                    self.scheduler.step(valid_loss)
                    
                    self.epoch_results['train_loss'].append(train_loss)
                    self.epoch_results['val_loss'].append(valid_loss)
                    self.epoch_results['val_acc'].append(valid_acc)
                    self.epoch_results['epoch'].append(epoch)
                    self.epoch_results['best_val_loss'].append(best_valid_loss)
                    self.epoch_results['best_val_acc'].append(best_valid_acc)
                    self.epoch_results['best_epoch'].append(best_epoch)
                    
                    
                    if best_valid_acc < valid_acc :
                        best_valid_loss = valid_loss
                        best_valid_acc = valid_acc
                        best_epoch = epoch                        
                        torch.save(self.model.state_dict(),f'{self.check_path}/{self.model_name}/{mainfold_idx}_{subfold_idx}_best_model.pt')
                        patience=0
                    else:
                        patience+=1
                        
#                     if epoch % 200 ==0:
#                         train_acc,train_loss = self.test(self.train_loader)
#                         test_acc,test_loss = self.test(self.test_loader) 

#                         self.overall_results['val_loss'].append(valid_loss)
#                         self.overall_results['val_acc'].append(valid_acc)
#                         self.overall_results['test_acc'].append(test_acc)
#                         self.overall_results['test_loss'].append(test_loss)
#                         self.overall_results['train_acc'].append(train_acc)
#                         self.overall_results['train_loss'].append(train_loss)
#                         self.overall_results['epoch'].append(epoch)

                    # if epoch % 200 ==0:
                    #     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
                    #     print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}')

                    if self.optimizer.param_groups[0]['lr'] <= minimum_lr or patience>100:                        
                        print(f'Mainfold_index: {mainfold_idx}, Subfold_index:{subfold_idx}')     
                        test_acc_final,test_loss_final = self.test(test_loader)   
                        torch.save(self.model.state_dict(),f'{self.check_path}/{self.model_name}/{mainfold_idx}_{subfold_idx}_final_model.pt')
                        self.model.load_state_dict(torch.load(f'{self.check_path}/{self.model_name}/{mainfold_idx}_{subfold_idx}_best_model.pt'))                        
                        test_acc,test_loss = self.test(test_loader)
                        
                        #results=pd.DataFrame(self.epoch_results)
                        #results.to_csv(f'{self.check_path}/{self.model_name}_{mainfold_idx}_{subfold_idx}_epoch_results.csv', na_rep='NaN')
                        # results2=pd.DataFrame(self.overall_results)
                        # results2.to_csv(f'{self.check_path}/{self.model_name}_{mainfold_idx}_{subfold_idx}_overall_results.csv', na_rep='NaN')

                        self.total_results['mainfold_index'].append(mainfold_idx)
                        self.total_results['subfold_index'].append(subfold_idx)
                        
                        self.total_results['best_epoch'].append(best_epoch)
                        self.total_results['best_val_loss'].append(best_valid_loss)
                        self.total_results['best_val_acc'].append(best_valid_acc)
                        
                        self.total_results['final_epoch'].append(epoch)
                        self.total_results['final_val_loss'].append(valid_loss)
                        self.total_results['final_val_acc'].append(valid_acc)
                        
                        self.total_results['best_test_loss'].append(test_loss)
                        self.total_results['best_test_acc'].append(test_acc)  
                        
                        self.total_results['final_test_loss'].append(test_loss_final)
                        self.total_results['final_test_acc'].append(test_acc_final)                         
                        
                        print(f'main & sub ==={mainfold_idx},{subfold_idx},best acc & loss==,{test_acc:.4f},{test_loss:.4f},final acc & loss=={test_acc_final:.4f},{test_loss_final:.4f},best_epoch=={best_epoch},final_epoch=={epoch}')
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
        
        results3=pd.DataFrame(self.total_results)
        results3.to_csv(f'{self.check_path}/{self.model_name}_{mainfold_idx}_total_results.csv', na_rep='NaN')
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
        with torch.no_grad():
            for data in loader:  # Iterate in batches over the training/test dataset.
                #print(len(data))
                data=data.to(self.device)
                out = self.model(data.x, data.edge_index, data.cycle_index, data.batch)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
                loss = self.criterion(out, data.y)
                total_loss+=loss.item()
        return correct / len(loader.dataset), total_loss/len(loader.dataset)  # Derive ratio of correct predictions.
                              
    def train_train(self):          
        self.model.train()
        total_loss=0
        for data in self.train_loader:  # Iterate in batches over the training dataset.
            self.optimizer.zero_grad()
            data=data.to(self.device)
            out = self.model(data.x, data.edge_index, data.cycle_index, data.batch)   # Perform a single forward pass.
            loss =self.criterion(out, data.y)  # Compute the loss.
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            total_loss+=loss.item()
        return total_loss/len(self.train_loader.dataset)          