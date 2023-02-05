import contextlib
import time
import torch
import numpy as np
import pandas as pd
from random import shuffle
import gudhi as gd
from scipy.sparse import coo_matrix
import itertools


def readFile(path):
    f = open(path)
    first_ele = True
    for data in f.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [int(x) for x in nums ]
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [int(x) for x in nums]
            matrix = np.c_[matrix,nums] #np.c_按行连接两个矩阵
    return matrix
def graph_to_simplex(edges,size,Max_dim):
    st=gd.SimplexTree()
    for v in range(size):
        st.insert([v])

    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)
        st.expansion(Max_dim)
    return st

def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]  #tuple:元组,特点是不可变

def build_tables(simplex_tree, size):
    complex_dim = simplex_tree.dimension()
    id_maps = [{} for _ in range(complex_dim+1)] # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim+1)] # matrix of simplices
    #boundaries_tables = [[] for _ in range(complex_dim+1)]
    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}
    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue
      # Assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)
    return simplex_tables, id_maps

def extract_boundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim):
    boundaries_tables = [[] for _ in range(complex_dim+1)]
    level_boundaries = {}
    for simplex, _ in simplex_tree.get_simplices():
        simplex_dim = len(simplex) - 1
        level_boundaries[tuple(simplex)]=list()
        if simplex_dim > 0:
            for boundary in get_simplex_boundaries(simplex):
                level_boundaries[tuple(simplex)].append(tuple(boundary)) #KeyError: (0, 1)
    return boundaries_tables, level_boundaries

def build_adj(simplex_table, id_maps,complex_dim,level_boundaries):
    adj_table = []
    for dim in range(1,complex_dim+1):
        simplex_id=[]
        boundary_id=[]
        values = []
        for id in range(len(simplex_table[dim])):
            #d=len(level_boundaries[tuple(simplex)])
            simplex=simplex_table[dim][id]
           # print("simplex",simplex)
            for j in range(len(level_boundaries[tuple(simplex)])):
                values.append((-1) ** j)
                #values.append(1)
                boundary=level_boundaries[tuple(simplex)][j]
                simplex_id.append(id_maps[dim][tuple(simplex)])
                boundary_id.append(id_maps[dim - 1][tuple(boundary)])
        adj_table.append(coo_matrix((values, (boundary_id, simplex_id)), shape=(len(simplex_table[dim-1]),len(simplex_table[dim]))).toarray())
    return adj_table
def build_laplacian(adj_table):
    laplacians = list()
   # up = coo_matrix(adj_table[0] @ adj_table[0].T)
    up = np.array(adj_table[0] @ adj_table[0].T)
    laplacians.append(up)
    for d in range(len(adj_table)-1):
        down = np.array(adj_table[d].T @ adj_table[d])
        up =np.array(adj_table[d + 1] @ adj_table[d + 1].T)
        laplacians.append(down + up)
    down = np.array(adj_table[-1].T @ adj_table[-1])
    laplacians.append(down)
    return laplacians
def construct_features(feature, simplex_tables,id_maps,feature_dim):
    features = list()
    features.append(feature.numpy())
    for dim in range(1,len(simplex_tables)):
       feature_d = np.zeros((len(simplex_tables[dim]), feature_dim))

       for c, cell in enumerate(simplex_tables[dim]): #c是index，cell是index对应的元素
            #for i in range(len(cell)):
               # print(cell_tables[dim])
                for _, node in enumerate(cell):  #c是index，cell是index对应的元素
                  feature_d[id_maps[dim][tuple(cell)]] = np.maximum(feature_d[id_maps[dim][tuple(cell)]],feature[int(node)])
       features.append(feature_d)
    return features
def construct_W(d,m,t):  #构建的w应该要满足高斯分布
    W=[[]]
    W.append(np.random.normal(d,m))
    for i in range(t-1):
        W.append(np.random.normal(m,m))
    return W
def SimHash(ts):
    threshold = 0
    ts = ts > threshold
    return  ts
def ComputeFingerprint(Max_dim,graph_id,laplacians,feature,simplex_tables,m,t,id_maps,per_dim,emb_dim,result_square):
    np.random.seed(229)
    if Max_dim==0:
        complex_dim=1
    else:
        complex_dim=len(simplex_tables)
    Htmp=[]
    Hin = construct_features(feature, simplex_tables,id_maps,feature.shape[1])
    for i in range(t): #迭代次数
        for d in range(complex_dim):
            # if i==0:
            #     result_tmp=np.matmul(laplacians[d],Hin[d])
            #     if result_tmp.shape[0]<per_dim[d]:
            #         tmp=np.zeros((per_dim[d]-result_tmp.shape[0],result_tmp.shape[1]))
            #         result_tmp=np.concatenate((result_tmp,tmp),0)
            #     else:
            #         result_tmp=result_tmp[:per_dim[d],]
            #     W = np.random.randn(result_tmp.shape[1],m)
            #     result_tmp=np.matmul(result_tmp,W)
            #     # file.write(str(graph_id) + str(d)+'\n')
            #     # str_w=[str(a) for a in W]
            #     # file.write('\n'.join(str_w)+'\n')
            #     # result_tmp = torch.mm(laplacians[d].float(), Hin[d].float())
            #     # tmp_w=torch.randn(Hin[d].shape[1]).unsqueeze(1)
            #     # result_tmp1=torch.mm(result_tmp,tmp_w).squeeze(1)
            #     # if(len(simplex_tables[d])>per_dim[d]):
            #     #     values, indices=result_tmp1.topk(per_dim[d],largest = False)
            #     #     result=result_tmp[indices,:]
            #     # else:
            #     #     values, indices = result_tmp1.topk(len(simplex_tables[d]), largest=False)
            #     #     result=torch.zeros((per_dim[d],Hin[d].shape[1]))
            #     #     result[:len(simplex_tables[d])]=result_tmp[indices,:]
            #     #result_tmp = result_tmp.view(-1).unsqueeze(dim=0)
            #     #W=torch.randn(feature.shape[1],m)
            #     #result=torch.mm(result,W)
            #     Htmp.append(SimHash(result_tmp))
            # else:
            #     W=np.random.randn(m,m)
            #     Htmp[d] = np.matmul(Htmp[d],W)
            #     Htmp[d] = SimHash(Htmp[d])
            # print(laplacians[d].shape,Hin[d].shape)
            result_tmp = np.matmul(laplacians[d], Hin[d])
            W = np.random.randn(result_tmp.shape[1], m)
            result = np.matmul(result_tmp, W)
            # Htmp.append(result_tmp)
            Hin[d] = SimHash(result)
    for d in range(complex_dim):
        if Hin[d].shape[0]<per_dim[d]:
            tmp=np.zeros((per_dim[d]-Hin[d].shape[0],Hin[d].shape[1]))
            Hin[d]=np.concatenate((Hin[d],tmp),0)
        else:
            Hin[d]=Hin[d][:per_dim[d],]
        tmp = Hin[d].flatten()
        if (d == 0):
            result = tmp
        else:
            result = np.concatenate((result, tmp), 0)
    result_square[i,graph_id,:len(result)]=result #i是迭代次数，graph_id是图

    return result_square

def embbeding(graph_id,datasets,edge_index,feature,Max_Dim,m,t,per_dim,emb_dim,result_square):
    size = feature.shape[0]
    st=graph_to_simplex(edge_index, size,Max_Dim)
    #print(dir(st))
    complex_dim=st.dimension()
    #np.random.seed(222)
    # print(complex_dim)
    simplex_tables, id_maps = build_tables(st, size)
    # if complex_dim >= 2:
    #     print(datasets,len(id_maps[complex_dim]))
    boundaries_tables, level_boundaries=extract_boundaries_from_simplex_tree(st,id_maps,size)
    adj_table=build_adj(simplex_tables, id_maps,complex_dim,level_boundaries)
    if(len(adj_table)==0):
        return False
    laplacians=build_laplacian(adj_table)
    result=ComputeFingerprint(Max_Dim,graph_id,laplacians,feature,simplex_tables,m,t,id_maps,per_dim,emb_dim,result_square)
    return True
    #end_time=time.time()


