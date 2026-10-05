"""Existing Week 4 toy primitive; not a complete GCN or GAT layer."""

import torch
from torch_geometric.nn import MessagePassing

class ProjectedSum(MessagePassing):
    def __init__(self,in_features,out_features):
        super().__init__(aggr=None,flow='source_to_target',node_dim=0)
        self.projection = torch.nn.Linear(in_features,out_features,bias=False)

    def forward(self,x,edge_index):
        h = self.projection(x)
        return self.propagate(edge_index,h=h,size=(x.shape[0],x.shape[0]))

    def message(self,h_j):
        return h_j

    def aggregate(self,inputs,index,ptr=None,dim_size=None):
        return inputs.new_zeros((dim_size,inputs.shape[1])).index_add(0,index,inputs)
