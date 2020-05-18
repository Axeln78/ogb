import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        
        if dropout != 0.0:
            self.drop_h = nn.Dropout(dropout)

    def forward(self, g, h, e, snorm_n, snorm_e):
        
        h_in = h  # for residual connection
        
        h = self.A(h) 
        
        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization  
        
        h = F.relu(h)  # non-linear activation
        
        if self.residual:
            h = h_in + h  # residual connection
            
        if self.dropout != 0:
            h = self.drop_h(h)  # dropout  
        
        return h, e


class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, graph_norm=True, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        
        if dropout != 0.0:
            self.drop_h = nn.Dropout(dropout)
            self.drop_e = nn.Dropout(dropout)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.data['Ce'] +  edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )
        return {'h' : h}
    
    def forward(self, g, h, e, snorm_n, snorm_e):
        
        h_in = h  # for residual connection
        e_in = e  # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 
        
        g.update_all(self.message_func, self.reduce_func) 
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution
        
        if self.graph_norm:
            h = h * snorm_n  # normalize activation w.r.t. graph size
            e = e * snorm_e  # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization  
            e = self.bn_node_e(e)  # batch normalization  
        
        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation
        
        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection
            
        if self.dropout != 0:
            h = self.drop_h(h)  # dropout  
            e = self.drop_e(e)  # dropout
        
        return h, e


class GNN(nn.Module):
    
    def __init__(self, gnn_type, num_tasks, num_layer=4, emb_dim=256, 
                 dropout=0.0, graph_norm=True, batch_norm=True, 
                 residual=True, graph_pooling="mean"):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.graph_pooling = graph_pooling
        
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        
        gnn_layer = {
            'gated-gcn': GatedGCNLayer,
            'Cheb_net': ChebLayer,
            'mlp': MLPLayer,
        }.get(gnn_type, GatedGCNLayer)
         
        self.layers = nn.ModuleList([
            gnn_layer(emb_dim, emb_dim, dropout=dropout, graph_norm=graph_norm, batch_norm=batch_norm, residual=residual) 
                for _ in range(num_layer) 
        ])
        
        self.pooler = {
            "mean": dgl.mean_nodes,
            "sum": dgl.sum_nodes,
            "max": dgl.max_nodes,
        }.get(graph_pooling, dgl.mean_nodes)
        
        self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
        
    def forward(self, g, h, e, snorm_n, snorm_e):
        
        h = self.atom_encoder(h)
        e = self.bond_encoder(e)
        
        for conv in self.layers:
            h, e = conv(g, h, e, snorm_n, snorm_e)
        g.ndata['h'] = h
        
        hg = self.pooler(g, 'h')
        
        return self.graph_pred_linear(hg)

    

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, k, activation, bias=True):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(k * in_feats, out_feats, bias)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, k):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(k * in_feats, out_feats)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}


class ChebLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, input_dim, output_dim, dropout=0.0, graph_norm=True, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self._k = 3

        if self.in_channels != self.out_channels:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(self.out_channels)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.apply_mod = NodeApplyModule(
            self.in_channels,
            self.out_channels,
            k=self._k)

    #def forward(self, g, feature, snorm_n, lambda_max=None):
    def forward(self, g, feature, e, snorm_n, snorm_e, lambda_max=[2]*128):
        h_in = feature   # to be used for residual connection

        def unnLaplacian(feature, D_sqrt, graph):
            """ Operation D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feature * D_sqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h') * D_sqrt

        with g.local_scope():
            D_sqrt = torch.pow(g.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feature.device)

            if lambda_max is None:
                try:
                    lambda_max = dgl.laplacian_lambda_max(g)
                except BaseException:
                    # if the largest eigonvalue is not found
                    lambda_max = [2]

            if isinstance(lambda_max, list):
                lambda_max = torch.Tensor(lambda_max).to(feature.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = dgl.broadcast_nodes(g, lambda_max)

            # X_0(f)
            Xt = X_0 = feature

            # X_1(f)
            if self._k > 1:
                re_norm = (2. / lambda_max).to(feature.device)
                h = unnLaplacian(X_0, D_sqrt, g)
                # print('h',h,'norm',re_norm,'X0',X_0)
                X_1 = - re_norm * h + X_0 * (re_norm - 1)

                Xt = torch.cat((Xt, X_1), 1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                h = unnLaplacian(X_1, D_sqrt, g)
                X_i = - 2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0

                Xt = torch.cat((Xt, X_i), 1)
                X_1, X_0 = X_i, X_1

            # Put the Chebyschev polynomes as featuremaps
            g.ndata['h'] = Xt
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata.pop('h')

        if self.graph_norm:
            h = h * snorm_n  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h, e