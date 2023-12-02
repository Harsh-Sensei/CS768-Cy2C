import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv,GINConv,Linear,LayerNorm
from torch_geometric.nn import global_mean_pool,global_add_pool,BatchNorm,global_max_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    
class Cy2C_GCN_OGB_1(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,p_ini=0.0,drop_mid=0.0):
        super(Cy2C_GCN_OGB_1, self).__init__()
        torch.manual_seed(12345)      
        self.emb = AtomEncoder(emb_dim = hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(self.n_layer)])
        self.conv_layers_2 = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(1)])
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(1)])
        self.lin1 = nn.Linear(hidden_channels*2,out_dim)
        self.act= nn.ReLU()        
        self.dropout = nn.Dropout(dropout_ratio)
        self.dropout2=nn.Dropout(drop_mid)
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
        aa=global_mean_pool(x_out_a,batch)    
        bb=global_mean_pool(x_out_b,batch)

        out=torch.cat((aa,bb),1)
        out=(self.lin1(out))
        return out  

class Cy2C_GCN_OGB_3(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,p_ini=0.0,drop_mid=0.0):
        super(Cy2C_GCN_OGB_3, self).__init__()
        torch.manual_seed(12345)      
        self.emb = AtomEncoder(emb_dim = hidden_channels)
        self.n_layer=n_layer
        self.conv_layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(self.n_layer)])
        self.conv_layers_2 = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(1)])
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(1)])
        self.lin1 = nn.Linear(hidden_channels*2,hidden_channels)
        self.lin2 = nn.Linear(hidden_channels,hidden_channels)
        self.lin3 = nn.Linear(hidden_channels,out_dim)
        
        self.act= nn.ReLU()       
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.dropout2=nn.Dropout(drop_mid)
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
        aa=global_mean_pool(x_out_a,batch)    
        bb=global_mean_pool(x_out_b,batch)

        out=torch.cat((aa,bb),1)
        out=self.lin3(self.act(self.lin2(self.act(self.lin1(out)))))
        return out    
    
class Cy2C_GCN_OGB_VN_attr(torch.nn.Module):
    def __init__(self, hidden_channels,in_dim,out_dim,dropout_ratio=0.0,n_layer=3,p_ini=0.0,drop_mid=0.0):
        super(Cy2C_GCN_OGB_VN_attr, self).__init__()
        torch.manual_seed(12345)      
        self.emb = AtomEncoder(emb_dim = hidden_channels)
        self.n_layer=n_layer
        self.bond=BondEncoder(emb_dim = 1)

        self.conv_layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels, normalize=False) for _ in range(self.n_layer)])
        self.conv_layers_2 = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(1)])
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(self.n_layer)])
        self.bn_layers_2 = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(1)])
        self.lin1 = nn.Linear(hidden_channels*2,out_dim)
        
        self.act= nn.ReLU()       
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.dropout2=nn.Dropout(drop_mid)
        self.VN_embedding = nn.Embedding(1, hidden_channels)
        torch.nn.init.constant_(self.VN_embedding.weight.data, 0)
        self.mlp_VN_list = nn.ModuleList()
        for layer in range(self.n_layer - 1):
            self.mlp_VN_list.append(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), BatchNorm(hidden_channels), 
                                                                 nn.ReLU(), nn.Linear(hidden_channels, hidden_channels), 
                                                                 BatchNorm(hidden_channels), torch.nn.ReLU()))        
        
    def forward(self, x, edge_index, cycle_index,  batch,edge_attr):
        # 1. Obtain node embeddings 
        edge_attr=self.bond(edge_attr)
        VN0=self.VN_embedding(torch.zeros(batch[-1].item()+1).to(edge_index.dtype).to(edge_index.device))        
        x_0=self.dropout(self.emb(x))
        a=[]
        x=x_0
        for i in range(self.n_layer):
            x=x+VN0[batch]
            xa=self.act(self.bn_layers[i](self.conv_layers[i](x,edge_index,edge_attr)))
            x=(self.dropout2(xa)+x)
            a.append(x)  
            if i < self.n_layer-1:
                vna=global_add_pool(x,batch) + VN0
                VN0 = VN0 + self.dropout2(self.mlp_VN_list[i](vna))
            
        x_out_a=x        
        x=x_0
        for i in range(1):
            xb=self.act(self.bn_layers_2[i](self.conv_layers_2[i](x,cycle_index)))
            x=(self.dropout2(xb)+x)
            a.append(x)
            
        x_out_b=x                
        aa=global_mean_pool(x_out_a,batch)    
        bb=global_mean_pool(x_out_b,batch)

        out=torch.cat((aa,bb),1)
        out=(self.lin1(out))
        return out        