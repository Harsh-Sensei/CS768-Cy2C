import numpy as np
# In[3]:
import torch.nn as nn
import torch_geometric.transforms as T
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree

def make_cycle_adj_speed_power(raw_list_adj,data):
    
    #original_adj=np.array(list(g.get_adjacency()))
    original_adj = np.array(raw_list_adj)
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=Xgraph.number_of_edges() - Xgraph.number_of_nodes() + nx.number_connected_components(Xgraph)
    node_each_cycle=nx.cycle_basis(Xgraph)
    if num_g_cycle >0 : #전체 그래프에서 단 1개의 사이클도 없을수도 있음.
  
        if len(node_each_cycle) != num_g_cycle:
            print('Error in the number of cycles in graph')
            print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=np.zeros((original_adj.shape[0],original_adj.shape[1]))
        for nodes in node_each_cycle:
            #start = time.time()
            #N_V=len(nodes)     
            #node 가 1, 5, 7, 10 에 만 있다.
            imsi=np.zeros((len(nodes),len(nodes)))
            #imsi = [4, 4]
            
            original_index=np.where(imsi!=0)
            for i,k in enumerate(nodes):
                imsi[i,:]=original_adj[k,nodes]/3
                imsi[i,i]=1/3
                
            if len(nodes)>1:
                 imsi=np.linalg.matrix_power(imsi,3)
                    
            #새로 만들어진 애들만 살림. 
            for i in range(len(nodes)):
                imsi[i,i]=0
            imsi[original_index[0],original_index[1]]=0
            
            #새로만들어진 애들에 3곱해
            #print(imsi, SUM_CYCLE_ADJ.shape, nodes)
            for i,k in enumerate(nodes):
                SUM_CYCLE_ADJ[k,nodes]=SUM_CYCLE_ADJ[k,nodes]+imsi[i,:]*3
            
        #전체에 오리지날 더해줘, 그러면 셀프 루프 1 빼고 완성!
   
            #SUM_CYCLE_ADJ[nodes,nodes]=0
            #print('3. time',time.time()-start)    
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=[]
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ, SUM_CYCLE_ADJ


def make_cycle_adj_speed_nosl(raw_list_adj,data):
    
    #original_adj=np.array(list(g.get_adjacency()))
    original_adj = np.array(raw_list_adj)
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=Xgraph.number_of_edges() - Xgraph.number_of_nodes() + nx.number_connected_components(Xgraph)
    node_each_cycle=nx.cycle_basis(Xgraph)
    if num_g_cycle >0 : #전체 그래프에서 단 1개의 사이클도 없을수도 있음.
  
        if len(node_each_cycle) != num_g_cycle:
            print('Error in the number of cycles in graph')
            print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=np.zeros((original_adj.shape[0],original_adj.shape[1]))
        for nodes in node_each_cycle:
            #start = time.time()
            #N_V=len(nodes)                
            for i in nodes:
                SUM_CYCLE_ADJ[i,nodes]=1   
            SUM_CYCLE_ADJ[nodes,nodes]=0
            #print('3. time',time.time()-start)    
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=[]
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ, SUM_CYCLE_ADJ

def make_cycle_adj_speed(raw_list_adj,data):
    
    #original_adj=np.array(list(g.get_adjacency()))
    original_adj = np.array(raw_list_adj)
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=Xgraph.number_of_edges() - Xgraph.number_of_nodes() + nx.number_connected_components(Xgraph)
    node_each_cycle=nx.cycle_basis(Xgraph)
    if num_g_cycle >0 : #전체 그래프에서 단 1개의 사이클도 없을수도 있음.
  
        if len(node_each_cycle) != num_g_cycle:
            print('Error in the number of cycles in graph')
            print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=np.zeros((original_adj.shape[0],original_adj.shape[1]))
        for nodes in node_each_cycle:
            #start = time.time()
            #N_V=len(nodes)                
            for i in nodes:
                SUM_CYCLE_ADJ[i,nodes]=1   
            #SUM_CYCLE_ADJ[nodes,nodes]=0
            #print('3. time',time.time()-start)    
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=[]
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ, SUM_CYCLE_ADJ

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def data_load(dataset, normalize=False,option=0):
    
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        print('max_degree',max_degree)
        
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    elif normalize:
        if option < 1 :
            mean = torch.mean(dataset.data.x, axis=0)
            std = torch.std(dataset.data.x, axis=0)
            dataset.data.x -= mean
            dataset.data.x /= std
        elif option >0:
            mean = torch.mean(dataset.data.x, axis=0)
            std = torch.std(dataset.data.x, axis=0)
            mean[option:]=0
            std[option:]=1
            dataset.data.x -= mean
            dataset.data.x /= std
        
    return dataset



def max_node_dataset(dataset):
    aa=[]
    for i in range(len(dataset)):
        data=dataset[i]
        aa.append(data.num_nodes)

    max_node=np.max(aa) #0~125번 노드까지 있으니까
    return max_node



# def make_igraph_g(_list_feature, list_adj):
#     _list_adj=list_adj.copy()
    
#     #바꿀땐 항상 주의
#     for i in range(_list_adj.shape[0]):
#         _list_adj[i,i]=0
        
#     _newg = ig.Graph.Adjacency(_list_adj, mode="undirected")    
#     _newg.vs['label']=_list_feature
#     for ii in  _newg.vs:     
#         ii["raw_id"]=ii.index        
#     return _newg


def make_cycle_adj(raw_list_adj,data):
    
    #original_adj=np.array(list(g.get_adjacency()))
    original_adj = np.array(raw_list_adj)
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=Xgraph.number_of_edges() - Xgraph.number_of_nodes() + nx.number_connected_components(Xgraph)
    node_each_cycle=nx.cycle_basis(Xgraph)
    if num_g_cycle >0 : #전체 그래프에서 단 1개의 사이클도 없을수도 있음.
  
        if len(node_each_cycle) != num_g_cycle:
            print('Error in the number of cycles in graph')
            print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=np.zeros((original_adj.shape[0],original_adj.shape[1]))
        for nodes in node_each_cycle:
    
            N_V=len(nodes)
            imsi_original_adj=original_adj            
            bool_adj=np.zeros((N_V,N_V))    
            for jj,i in enumerate(nodes):
                bool_adj[jj,:]=imsi_original_adj[i,nodes]
                bool_adj[:,jj]=imsi_original_adj[nodes,i]
                bool_adj[jj,jj]=1
            
            #print(N_V)
            bool_adj=bool_adj/3
            #print(bool_adj)
            if int(N_V-1)>1:
                cycle_adj_cal = np.linalg.matrix_power(bool_adj,int(N_V-1)) #이거 csr은 ** 인데, (csr은 파워가 element-wise임, numpy랑 반대임 ㅡㅡ)
            else:
                cycle_adj_cal = bool_adj         
                
            cycle_matrix=np.zeros((original_adj.shape[0],original_adj.shape[1]))
            
            for jj,i in enumerate(nodes):
                cycle_matrix[i,nodes]=cycle_adj_cal[jj,:]
                cycle_matrix[nodes,i]=cycle_adj_cal[:,jj]                
                
            #print(cycle_adj_cal)    
            #CAL_SUB_ADJ.append(cycle_matrix)    
            SUM_CYCLE_ADJ = SUM_CYCLE_ADJ + cycle_matrix
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=[]
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ, SUM_CYCLE_ADJ



def make_cycle_adj_cut(raw_list_adj,data):
    
    #original_adj=np.array(list(g.get_adjacency()))
    original_adj = np.array(raw_list_adj)
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=Xgraph.number_of_edges() - Xgraph.number_of_nodes() + nx.number_connected_components(Xgraph)
    node_each_cycle=nx.cycle_basis(Xgraph)
    new_node=[]
    for i in node_each_cycle:
        if len(i)>3:
            new_node.append(i)
            
    node_each_cycle=new_node
    
    if num_g_cycle >0 : #전체 그래프에서 단 1개의 사이클도 없을수도 있음.
  
        # if len(node_each_cycle) != num_g_cycle:
        #     print('Error in the number of cycles in graph')
        #     print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=np.zeros((original_adj.shape[0],original_adj.shape[1]))
        for nodes in node_each_cycle:
    
            N_V=len(nodes)
            imsi_original_adj=original_adj            
            bool_adj=np.zeros((N_V,N_V))    
            for jj,i in enumerate(nodes):
                bool_adj[jj,:]=imsi_original_adj[i,nodes]
                bool_adj[:,jj]=imsi_original_adj[nodes,i]
                bool_adj[jj,jj]=1
            
            #print(N_V)
            bool_adj=bool_adj/3
            #print(bool_adj)
            if int(N_V-1)>1:
                cycle_adj_cal = np.linalg.matrix_power(bool_adj,int(N_V-1)) #이거 csr은 ** 인데, (csr은 파워가 element-wise임, numpy랑 반대임 ㅡㅡ)
            else:
                cycle_adj_cal = bool_adj         
                
            cycle_matrix=np.zeros((original_adj.shape[0],original_adj.shape[1]))
            
            for jj,i in enumerate(nodes):
                cycle_matrix[i,nodes]=cycle_adj_cal[jj,:]
                cycle_matrix[nodes,i]=cycle_adj_cal[:,jj]                
                
            #print(cycle_adj_cal)    
            CAL_SUB_ADJ.append(cycle_matrix)    
            SUM_CYCLE_ADJ = SUM_CYCLE_ADJ + cycle_matrix
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=[]
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ, SUM_CYCLE_ADJ

def make_cycle_adj_old(g,raw_list_adj,data):
 
    original_adj=np.array(list(g.get_adjacency()))
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=g.ecount() - g.vcount() + len(g.components())
    node_each_cycle = len(nx.cycle_basis(Xgraph))
    node_each_cycle=nx.cycle_basis(Xgraph)
    if num_g_cycle >0 : #전체 그래프에서 단 1개의 사이클도 없을수도 있음.
  
        if len(node_each_cycle) != num_g_cycle:
            print('Error in the number of cycles in graph')
            print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        
        for k in node_each_cycle:
            N_V=len(k)
            imsi_original_adj=original_adj
            bool_adj=np.zeros((original_adj.shape[0],original_adj.shape[1]))
            for i in k:
                bool_adj[i,k]=1
                bool_adj[k,i]=1    
                imsi_original_adj[i,i]=1
            #self_loop 추가
                       
            imsi_original_adj= csr_matrix(imsi_original_adj)    
            
            bool_adj= csr_matrix(bool_adj)
            
            cycle_adj=bool_adj.multiply(original_adj)
            SUB_ADJ.append(cycle_adj) 
            
            cycle_raw_adj=bool_adj.multiply(imsi_original_adj)               
            RAW_SUB_ADJ.append(cycle_raw_adj)
            
            _cycle_adj_cal=cycle_raw_adj/3
            
            if int(N_V-1)>1:
                cycle_adj_cal = _cycle_adj_cal**(int(N_V-1)) #이거 csr은 ** 인데, (csr은 파워가 element-wise임, numpy랑 반대임 ㅡㅡ)
            else:
                cycle_adj_cal=_cycle_adj_cal

            CAL_SUB_ADJ.append(cycle_adj_cal)    
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ