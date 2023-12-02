import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv,GINConv,Linear,LayerNorm
from torch_geometric.nn import global_mean_pool,global_add_pool,BatchNorm
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,n_head=8,drop_mid=0.0, drop_ini=0.0):
        super(GAT, self).__init__()
        torch.manual_seed(12345)      
        self.mid_dim=int(hidden_channels/n_head)
        self.emb= Linear(in_dim, hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GATConv(hidden_channels, self.mid_dim ,heads=n_head) for _ in range(self.n_layer-1)])
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        #self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        
        #-----차이점
        self.conv_layers.append(GATConv(hidden_channels, hidden_channels,heads=1))
        #self.conv_layer_2=GATConv(hidden_channels, hidden_channels,heads=1)
        self.lin1 = nn.Linear(hidden_channels,hidden_channels)
        self.lin2 = nn.Linear(hidden_channels,hidden_channels)
        self.lin3 = nn.Linear(hidden_channels,out_dim)
        
        self.act= nn.ReLU()       
        
        self.dropout = nn.Dropout(drop_ini)
        self.dropout2= nn.Dropout(drop_mid)
    def forward(self, x, edge_index, cycle_index,  batch):
        # 1. Obtain node embeddings 
        
        x_0=self.dropout(self.emb(x))
        a=[]
        x=x_0
        for i in range(self.n_layer):
            xa=self.act(self.bn_layers[i](self.conv_layers[i](x,edge_index)))
            x=self.dropout2(xa)+x
            a.append(x)  
        x_out_a=x        
                
        out=global_mean_pool(x_out_a,batch)    

        out=self.lin3(self.act(self.lin2(self.act(self.lin1(out)))))
        return out
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,drop_mid=0.0, drop_ini=0.0):
        super(GCN, self).__init__()
        torch.manual_seed(12345)      
        self.emb= Linear(in_dim, hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(self.n_layer)])
        #self.conv_layers_2 = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(1)])
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        #self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.lin1 = nn.Linear(hidden_channels,hidden_channels)
        self.lin2 = nn.Linear(hidden_channels,hidden_channels)
        self.lin3 = nn.Linear(hidden_channels,out_dim)
        self.act= nn.ReLU()       
        self.dropout2=nn.Dropout(drop_mid)
        self.dropout = nn.Dropout(drop_ini)
        
    def forward(self, x, edge_index, cycle_index,  batch):
        # 1. Obtain node embeddings 

        x_0=self.dropout(self.emb(x))
        a=[]
        x=x_0
        for i in range(self.n_layer):
            xa=self.act(self.bn_layers[i](self.conv_layers[i](x,edge_index)))
            x=self.dropout2(xa)+x
            a.append(x)  
            
        x_out_a=x        
                
        out=global_mean_pool(x_out_a,batch)    

        out=self.lin3(self.act(self.lin2(self.act(self.lin1(out)))))
        return out
    
class Cy2C_GCN(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,drop_mid=0.0, drop_ini=0.0):
        super(Cy2C_GCN, self).__init__()
        torch.manual_seed(12345)      
        self.emb= Linear(in_dim, hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(self.n_layer)])
        self.conv_layers_2 = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(1)])
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.lin1 = nn.Linear(hidden_channels*2,hidden_channels)
        self.lin2 = nn.Linear(hidden_channels,hidden_channels)
        self.lin3 = nn.Linear(hidden_channels,out_dim)
        self.act= nn.ReLU()       
        self.dropout2=nn.Dropout(drop_mid)
        self.dropout = nn.Dropout(drop_ini)
        
    def forward(self, x, edge_index, cycle_index,  batch):
        # 1. Obtain node embeddings 

        x_0=self.dropout(self.emb(x))
        a=[]
        x=x_0
        for i in range(self.n_layer):
            xa=self.act(self.bn_layers[i](self.conv_layers[i](x,edge_index)))
            x=self.dropout2(xa)+x
            a.append(x)  
            
        x_out_a=x        
        x=x_0
        for i in range(1):
            xb=self.act(self.bn_layers_2[i](self.conv_layers_2[i](x,cycle_index)))
            x=self.dropout2(xb)+x
            a.append(x)           
        x_out_b=x
                
        aa=global_mean_pool(x_out_a,batch)    
        bb=global_mean_pool(x_out_b,batch)

        out=torch.cat((aa,bb),1)
        out=self.lin3(self.act(self.lin2(self.act(self.lin1(out)))))
        return out
    
class Cy2C_GAT(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,n_head=8,drop_mid=0.0, drop_ini=0.0):
        super(Cy2C_GAT, self).__init__()
        torch.manual_seed(12345)      
        self.mid_dim=int(hidden_channels/n_head)
        self.emb= Linear(in_dim, hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GATConv(hidden_channels, self.mid_dim ,heads=n_head) for _ in range(self.n_layer-1)])
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        
        #-----차이점
        self.conv_layers.append(GATConv(hidden_channels, hidden_channels,heads=1))
        self.conv_layer_2=GATConv(hidden_channels, hidden_channels,heads=1)
        self.lin1 = nn.Linear(hidden_channels*2,hidden_channels)
        self.lin2 = nn.Linear(hidden_channels,hidden_channels)
        self.lin3 = nn.Linear(hidden_channels,out_dim)
        
        self.act= nn.ReLU()       
        
        self.dropout = nn.Dropout(drop_ini)
        self.dropout2= nn.Dropout(drop_mid)
    def forward(self, x, edge_index, cycle_index,  batch):
        # 1. Obtain node embeddings 
        
        x_0=self.dropout(self.emb(x))
        a=[]
        x=x_0
        for i in range(self.n_layer):
            xa=self.act(self.bn_layers[i](self.conv_layers[i](x,edge_index)))
            x=self.dropout2(xa)+x
            a.append(x)  
            
        x_out_a=x        
        x=x_0
        for i in range(1):
            xb=self.act(self.bn_layers_2[i](self.conv_layer_2(x,cycle_index)))
            x=self.dropout2(xb)+x
            a.append(x)           
        x_out_b=x
                
        aa=global_mean_pool(x_out_a,batch)    
        bb=global_mean_pool(x_out_b,batch)

        out=torch.cat((aa,bb),1)
        out=self.lin3(self.act(self.lin2(self.act(self.lin1(out)))))
        return out

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,drop_mid=0.0,drop_ini=0.0):
        super(GIN, self).__init__()
        torch.manual_seed(12345)
 
        self.emb= Linear(in_dim, hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GINConv(
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                       nn.Linear(hidden_channels, hidden_channels))) for _ in range(self.n_layer)])
#         self.conv_layers_2 = nn.ModuleList([GINConv(
#             nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
#                        nn.Linear(hidden_channels, hidden_channels))) for _ in range(1)])                        
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.lins = nn.ModuleList([Linear(hidden_channels,out_dim) for _ in range(int(self.n_layer))])
#         self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(1)])
        self.act= nn.ReLU()       
        self.dropout2 = nn.Dropout(drop_mid)
        self.dropout = nn.Dropout(drop_ini)
        
    def forward(self, x, edge_index, cycle_index,  batch):
        # 1. Obtain node embeddings 

        x_0=self.dropout(self.emb(x))
        a=[]
        x=x_0
        for i in range(self.n_layer):
            xa=self.act(self.bn_layers[i](self.conv_layers[i](x,edge_index)))
            x=(self.dropout2(xa)+x)
            a.append(x)  
            
        
        out_over_layer=0
        for i in range(len(a)):
            pool_a=global_add_pool(a[i],batch)
            out_over_layer += self.lins[i](pool_a)

        return out_over_layer

class Cy2C_GIN(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,drop_mid=0.0,drop_ini=0.0):
        super(Cy2C_GIN, self).__init__()
        torch.manual_seed(12345)
 
        self.emb= Linear(in_dim, hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GINConv(
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                       nn.Linear(hidden_channels, hidden_channels))) for _ in range(self.n_layer)])
        self.conv_layers_2 = nn.ModuleList([GINConv(
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                       nn.Linear(hidden_channels, hidden_channels))) for _ in range(1)])                        
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.lins = nn.ModuleList([Linear(hidden_channels,out_dim) for _ in range(int(self.n_layer+1))])
        self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(1)])
        self.act= nn.ReLU()       
        self.dropout2 = nn.Dropout(drop_mid)
        self.dropout = nn.Dropout(drop_ini)
        
    def forward(self, x, edge_index, cycle_index,  batch):
        # 1. Obtain node embeddings 

        x_0=self.dropout(self.emb(x))
        a=[]
        x=x_0
        for i in range(self.n_layer):
            xa=self.act(self.bn_layers[i](self.conv_layers[i](x,edge_index)))
            x=(self.dropout2(xa)+x)
            a.append(x)  
            
        x_out_a=x        
        x=x_0
        for i in range(1):
            xb=self.act(self.bn_layers_2[i](self.conv_layers_2[i](x,cycle_index)))
            x=(self.dropout2(xb)+x)
            a.append(x)           
        x_out_b=x
        
        out_over_layer=0
        for i in range(len(a)):
            pool_a=global_add_pool(a[i],batch)
            out_over_layer += self.lins[i](pool_a)

        return out_over_layer