import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as M
import math
import torch.nn as nn
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
#from torch.hub import load_state_dict_from_url
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torchvision.models.resnet
import pdb
'''
==============================================================================
model_g, model_d, model_x, model_a, model_i2,
model_ga, model_da, model_gx, model_dx, 
==============================================================================
'''

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def gen_adj(A, adj_init):
    #print(A.sum(1).float())
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
    
def gen_A_extra(num_classes, num, t, adj_file, adj_relation, adj_norelation, extra_adj):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / (_nums+ 1e-8)
    print(t)
    print(num)
    _adj[_adj >= t] = 1#adj_relation#1

    _adj[_adj < t] = 0#adj_relation#1
    _adj = np.pad(_adj,  ((0,num), (0,num)), 'constant',  constant_values = (extra_adj,extra_adj))#0.0001

    row, col = np.diag_indices_from(_adj)
    _adj[row[-num:], col[-num:]] = 0
    
    #pdb.set_trace()
    _adj = _adj + np.identity(num_classes+num, np.int)


    return _adj
    

class GCNResnet_extra(nn.Module):
    def __init__(self, extra_num=0, t=0.9, num_class=30, adj_file=None, adj_relation=1, adj_norelation=0, adj_init=0.01, extra_adj = 0.00001):
        super(GCNResnet_extra, self).__init__()
        '''
        extra_num: the number of the extra attr
        t : threshold of adjacent matrix 
        num_class: the number of origin attr
        adj_file :the pickle file of adjacent
        adj_init: laplacian normalization add
        extra_adj: the adj matrix of the extra attrs
        '''
        self.gc1 = GraphConvolution(50, 512)
        self.gc2 = GraphConvolution(512, 512)
        self.gc3 = GraphConvolution(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU(0.2)
        
        self.adj_init = adj_init
        _adj = gen_A_extra(num_class, extra_num, t, adj_file, adj_relation, adj_norelation, extra_adj)
          
        self.A = nn.Parameter(torch.from_numpy(_adj).float(), requires_grad=True)#30 30

        self.extra_num = extra_num
        #self.extra_attr=[]
        if extra_num !=0 :
            self.extra_attr = torch.nn.Parameter(torch.FloatTensor(self.extra_num, 50), requires_grad=True)
            #torch.nn.init.normal(self.extra_attr, 0, 0.25)
            torch.nn.init.orthogonal_(self.extra_attr, gain=1)
            #self.extra_attr.data.fill_(0)
            #torch.nn.init.constant(self.extra_attr, 0.00001)

    '''
    def gc_params(self):
        stn = []
        #stn.extend([*self.gc1.parameters()])
        stn.extend([*self.gc2.parameters()])
        stn.extend([*self.gc3.parameters()])
        stn.extend([*self.bn3.parameters()])
        stn.extend([self.A])
      
        return stn  
    '''
   
    def attr_params(self):
        if self.extra_num==0:
            return []
        return [self.extra_attr]

        
    def forward(self, inp):
        
        #pdb.set_trace()
        #adj =  gen_adj(( self.A - torch.min(self.A, dim=1)[0] )/( torch.max(self.A, dim=1)[0] - torch.min(self.A, dim=1)[0] ))#.detach()
        #adj = gen_adj(torch.relu(self.A))#.detach()
        #print(self.extra_attr)
        
        #print("**************adjacent***************")
        #print(self.A[30:,:])
        adj = gen_adj(torch.relu(self.A), 0)#.detach()
        #print("************0.01************")
        #print(adj[:5,:])
        '''
        adj = gen_adj(torch.relu(self.A), 0)#.detach()
        print("************0************")
        print(adj[:5,:])
        adj = gen_adj(torch.relu(self.A), 1e-12)#.detach()
        print("************1e-12************")
        print(adj[:5,:])        
        '''
        #adj = torch.relu(self.A)
        #print(adj[0])
        '''
        adj[adj>=0.9]=1
        adj[adj<0.9]=0
        ones = torch.ones(30,1).cuda()
        index = torch.arange(30).cuda().unsqueeze(1)
        adj.scatter_(1,index,ones) 
        #pdb.set_trace()
        adj = gen_adj(adj)
        #print(torch.sum(adj))
        '''
        if self.extra_num!=0:
            new_attr = self.extra_attr#torch.stack(self.extra_attr, dim=0).cuda()#60,50
            inp = torch.cat((inp, new_attr.unsqueeze(0).expand((inp.size(0), self.extra_num, 50)) ),dim=1)#bs 90 50
        #pdb.set_trace()
        
          
        x = self.gc1(inp, adj)
       
        x = F.relu(x)
        x = self.gc3(x, adj)         
       
        #x = torch.mean(x, 1)
        x = torch.max(x, 1)[0]
        
        x = self.bn3(x)

        return x

class model_g(nn.Module):
    def __init__(self, input_size, output_size):

        super(model_g, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, output_size)

    def forward(self, feature):

        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        f = self.linear4(f)
        output = torch.tanh(f)

        return output
        
class model_glove(nn.Module):
    def __init__(self):

        super(model_glove, self).__init__()
        self.linear1 = nn.Linear(50, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, 512)

    def forward(self, feature):
        #pdb.set_trace()
        bs  = feature.size()[0]
        feature = feature.reshape(-1,50)
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        output = self.linear4(f)
        output = output.reshape(bs, -1,512)
        #output = torch.mean(output,dim=1)
        output = torch.max(output, 1)[0]
        output = torch.tanh(output)

        return output
        
class model_glove_extra(nn.Module):
    def __init__(self, num_attr):

        super(model_glove_extra, self).__init__()
        self.linear1 = nn.Linear(50, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, 512)  
        self.extra_num = num_attr
        self.extra_attr = torch.nn.Parameter(torch.FloatTensor(self.extra_num, 50), requires_grad=True)
        #torch.nn.init.normal(self.extra_attr, 0, 0.25)
        torch.nn.init.orthogonal_(self.extra_attr, gain=1)
    
    def forward(self, feature):
        #pdb.set_trace()
        bs  = feature.size()[0]
        feature = feature.reshape(bs,-1,50)#bs x attr, 50
        feature = torch.cat((feature, self.extra_attr.unsqueeze(0).expand((bs, self.extra_num, 50)) ),dim=1)#bs 90 50
        feature = feature.reshape(-1, 50)
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        output = self.linear4(f)
        output = output.reshape(bs, -1,512)
        output = torch.max(output, 1)[0]
        output = torch.tanh(output)

        return output        


# discriminator for the attribute branch


class model_d(nn.Module):

    def __init__(self, input_size):
        super(model_d, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear3(f)
        output = torch.sigmoid(f)

        return output


# encoder for the image branch


class model_x(nn.Module):

    def __init__(self, input_size, output_size):
        super(model_x, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, output_size)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        output = torch.tanh(f)

        return output


# encoder for the attribute branch


class model_a(nn.Module):

    def __init__(self, input_size, output_size):
        super(model_a, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, output_size)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        output = torch.tanh(f)

        return output


# shared classifier


class model_i2(nn.Module):

    def __init__(self, input_size, output_size):
        super(model_i2, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)

    def forward(self, feature):
        output = self.linear1(feature)

        return output


# generator for the fake attribute features a'


class model_ga(nn.Module):
    def __init__(self, input_size, output_size):

        super(model_ga, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, output_size)

    def forward(self, feature):

        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        f = self.linear4(f)
        output = torch.tanh(f)

        return output


# generator for the fake image features x'


class model_gx(nn.Module):
    def __init__(self, input_size, output_size):

        super(model_gx, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, output_size)

    def forward(self, feature):

        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        f = self.linear4(f)
        output = torch.tanh(f)

        return output


# discriminator for a' and a


class model_da(nn.Module):

    def __init__(self, input_size):
        super(model_da, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear3(f)
        output = torch.sigmoid(f)

        return output


# discriminator for x' and x


class model_dx(nn.Module):

    def __init__(self, input_size):
        super(model_dx, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear3(f)
        output = torch.sigmoid(f)

        return output