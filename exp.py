from torch_geometric.datasets import TUDataset
from utils import embbeding
import time
import torch
import numpy as np
from scipy.spatial.distance import pdist,squareform
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
np.set_printoptions(threshold=np.inf)
dataset='FRANKENSTEIN'
datasets = TUDataset(root=f'/tmp/{dataset}', name=f'{dataset}')

per_dim=[15,20,50,20]
#DHFR:100 1 1
#PRO: 50 1 2
def main():
    M=[25,50,100,150]
    T=[1,2,3,4,5]
    max_dim=2
    # m = int(input())  #特征维度
    # t = int(input())   #迭代次数
    # max_dim=int(input())
    Acc=[]
    Cost=[]
    for m in M:
        for t in T:
            start_time = time.time()
            classes=[]
            result_square = np.zeros((t, len(datasets), m*(sum(per_dim[:max_dim+1]))))
            for i in range(len(datasets)):
                data=datasets[i]
                result_square=embbeding(i,dataset,data.edge_index,data.x, max_dim,m,t,per_dim,m*(sum(per_dim[:max_dim+1])),result_square)
                classes.append(int(data.y))
            gram_matrix=np.zeros((len(datasets),len(datasets)))
            for r in range(t):
                gram_matrix=gram_matrix+(1-squareform(pdist(result_square[r], 'hamming')))
            predictor = svm.SVC(C=2, kernel='precomputed')
            scores = cross_val_score(predictor, gram_matrix, classes, scoring='f1')
            print("Score: {}".format(np.mean(scores)))
            end_time = time.time()
            print("程序的运行时间为",(end_time-start_time))
            Acc.append(np.mean(scores))
            Cost.append(end_time-start_time)
            msg = (
                f'========== Result ============\n'
                f'Dataset:        {dataset}\n'
                f'Accuracy:       {np.mean(scores)}\n'
                f'Cost_time:      {end_time-start_time}\n'
                '-------------------------------\n\n')
            file = open(f'/home/xuantan/My_projects/cwn-main/scnn_tx/results/{dataset}/{dataset}_M{str(m)}_T{str(t)}.txt', 'w')
            file.write(msg)
    Final_msg= (
        f'========== Final Result ============\n'
        f'Dataset:             {dataset}\n'
        f'Best Accuracy:       {max(Acc)}\n'
        f'Min Cost_time:       {min(Cost)}\n'
        f'Var Cost_time:       {np.var(Cost)}\n'
        '-------------------------------\n\n')
    f= open(f'/home/xuantan/My_projects/cwn-main/scnn_tx/results/{dataset}/{dataset}_final_result.txt', 'w')
    f.write(Final_msg)


main()

