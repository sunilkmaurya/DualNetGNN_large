import pickle
import numpy as np
import torch
import time
from dataset import load_nc_dataset
import torch_sparse
from data_utils import gen_normalized_adjs, gen_normalized_adjs_i, load_fixed_splits
from torch_geometric.utils import to_undirected

data_loc = "./data/"
datasets = [('genius',''),('twitch-gamer',''),('arxiv-year',''),('fb100','Penn94'),('pokec',''),('snap-patents','')]
list_num_hops_adj = [0,0,3,0,0,0]
list_num_hops_feat = [4,4,3,4,4,3]
list_self_loop_adj = [True,True,True,True,True,True]
list_no_loop_adj = [True,True,False,True,True,True]
list_self_loop_feat = [True,True,True,True,True,True]
list_no_loop_feat = [True,True,False,True,True,True]

#Loading or pre-processing function

for ind, db in enumerate(datasets):
    print("=========================")
    print(f"Processing {db[0]}...")
    dataset = load_nc_dataset(db[0],db[1])

    num_hops_adj = list_num_hops_adj[ind]
    num_hops_feat = list_num_hops_feat[ind]
    is_self_loop_adj = list_self_loop_adj[ind]
    is_no_loop_adj = list_no_loop_adj[ind]
    is_self_loop_feat = list_self_loop_feat[ind]
    is_no_loop_feat = list_no_loop_feat[ind]

    if db[0] in ['snap-patents','arxiv-year']:
        as_directed = True
    else:
        as_directed = False

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if  db in ['ogbn-proteins', 'wiki']:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                    for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(db[0], db[1])


    num_nodes = dataset.graph['num_nodes']
    # infer the number of classes for non one-hot and one-hot labels
    num_label = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    num_feat = dataset.graph['node_feat'].shape[1]

    #If undirected, matrices are A, (A+I)
    #If directed, matrices are A,(A+I), A^T, (A+I)^T
    if as_directed == True:
        #graph is processed as directed
        print(f"Creating directed matrices")
        _,adj_i = gen_normalized_adjs_i(dataset)
        _,adj = gen_normalized_adjs(dataset)
        _,adj_t_i = gen_normalized_adjs_i(dataset,transpose=True)
        _,adj_t = gen_normalized_adjs(dataset,transpose=True)
    else:
        #graph is processed as undirected
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        adj_i, _ = gen_normalized_adjs_i(dataset)
        adj, _ = gen_normalized_adjs(dataset)


    list_mat = []
    if is_no_loop_adj:
        list_mat.append(adj)
    if is_self_loop_adj:
        list_mat.append(adj_i)


    if is_no_loop_adj:
        tmp_mat = adj
        for i in range(num_hops_adj):
            tmp_mat = tmp_mat.spspmm(adj)
            list_mat.append(tmp_mat)

    if is_self_loop_adj:
        tmp_mat = adj_i
        for i in range(num_hops_adj):
            tmp_mat = tmp_mat.spspmm(adj)
            list_mat.append(tmp_mat)

    if as_directed == True:
        print(f"Also adding transpose adajcency")
        if is_no_loop_adj:
            list_mat.append(adj_t)
        if is_self_loop_adj:
            list_mat.append(adj_t_i)

        if is_no_loop_adj:
            tmp_mat = adj_t
            for i in range(num_hops_adj):
                tmp_mat = tmp_mat.spspmm(adj_t)
                list_mat.append(tmp_mat)

        if is_self_loop_adj:
            tmp_mat = adj_t_i
            for i in range(num_hops_adj):
                tmp_mat = tmp_mat.spspmm(adj_t)
                list_mat.append(tmp_mat)




    loop_feat = dataset.graph['node_feat']
    no_loop_feat = dataset.graph['node_feat']

    #Adding nodes features to the list
    list_mat.append(dataset.graph['node_feat'])

    for i in range(num_hops_feat):
        if is_no_loop_feat:
            no_loop_feat = adj.spmm(no_loop_feat)
            list_mat.append(no_loop_feat)
        if is_self_loop_feat:
            loop_feat = adj_i.spmm(loop_feat)
            list_mat.append(loop_feat)

    if as_directed == True:
        loop_feat = dataset.graph['node_feat']
        no_loop_feat = dataset.graph['node_feat']
        for i in range(num_hops_feat):
            if is_no_loop_feat:
                no_loop_feat = adj_t.spmm(no_loop_feat)
                list_mat.append(no_loop_feat)
            if is_self_loop_feat:
                loop_feat = adj_t_i.spmm(loop_feat)
                list_mat.append(loop_feat)

    print(f"Total number of feature matrices: {len(list_mat)}")
    with open("./processed_data/"+db[0]+".pickle","wb") as fopen:
        pickle.dump([list_mat, dataset.label, split_idx_lst, num_nodes, num_feat,num_label], fopen)
    print(f"Files saved.")
