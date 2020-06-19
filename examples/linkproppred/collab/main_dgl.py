import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

from logger import Logger
from torch.utils.tensorboard import SummaryWriter


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

    def forward(self, g, h, e):
        
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


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        
        if dropout != 0.0:
            self.drop_h = nn.Dropout(dropout)

    def message_func(self, edges):
        Bh_j = edges.src['Bh'] 
        return {'Bh_j' : Bh_j}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        h = Ah_i + torch.sum( Bh_j, dim=1 )
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h  # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        
        g.update_all(self.message_func, self.reduce_func) 
        h = g.ndata['h']  # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization  
        
        h = F.relu(h)  # non-linear activation
        
        if self.residual:
            h = h_in + h  # residual connection
            
        if self.dropout != 0:
            h = self.drop_h(h)  # dropout  
        
        return h, e
    

class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
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
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )
        return {'h' : h}
    
    def forward(self, g, h, e):
        
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
    
    def __init__(self, gnn_type="gated_gcn", in_dim=128, in_dim_edge=1, emb_dim=64, 
                 num_layer=3, dropout=0.0, batch_norm=True, residual=True, edge_feat=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.in_dim_edge = in_dim_edge
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.edge_feat = edge_feat
        
        self.embedding_h = nn.Linear(in_dim, emb_dim, bias=True)
        self.embedding_e = nn.Linear(in_dim_edge, emb_dim, bias=True)
        
        gnn_layer = {
            'gated-gcn': GatedGCNLayer,
            'gcn': GCNLayer,
            'mlp': MLPLayer
        }.get(gnn_type, GatedGCNLayer)
         
        self.layers = nn.ModuleList([
            gnn_layer(emb_dim, emb_dim, dropout=dropout, batch_norm=batch_norm, residual=residual) 
                for _ in range(num_layer) 
        ])
        
    def forward(self, g, h, e):
        
        h = self.embedding_h(h)
        
        if not self.edge_feat:
            e = torch.ones_like(e).to(e.device)
        e = self.embedding_e(e)
        
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        g.edata['e'] = e
        
        return h
    

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

    
class LinkPredictor(nn.Module):
    def __init__(self, emb_dim=64, out_dim=1, num_layers=2, dropout=0.0):
        super(LinkPredictor, self).__init__()
        
        self.MLP_layer = MLPReadout(2 * emb_dim, out_dim, num_layers)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        x = self.MLP_layer(x)
        return torch.sigmoid(x)
    

def train(model, predictor, data, split_edge, optimizer, batch_size, device):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(device)

    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)):
        
        optimizer.zero_grad()
        
        g = data.to(device)
        x = g.ndata['feat'].to(device)
        e = g.edata['edge_weight'].to(device).float()
        
        h = model(g, x, e)
        
        edge = pos_train_edge[perm].t()

        pos_out = predictor( h[edge[0]], h[edge[1]] )
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=x.device)
        
        neg_out = predictor( h[edge[0]], h[edge[1]] )
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, device):
    model.eval()
    predictor.eval()

    g = data.to(device)
    x = g.ndata['feat'].to(device)
    e = g.edata['edge_weight'].to(device).float()

    h = model(g, x, e)

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    
    parser.add_argument('--gnn_type', type=str, default='gcn')
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    parser.add_argument('--batch_size', type=int, default=32*1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    print(args)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)

    dataset = DglLinkPropPredDataset(name='ogbl-collab')
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    print(data)

    model = GNN(gnn_type=args.gnn_type, emb_dim=args.emb_dim, num_layer=args.num_layer, dropout=args.dropout).to(device)
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Model parameters: {total_param}')

    predictor = LinkPredictor(emb_dim=args.emb_dim).to(device)
    print(predictor)
    total_param = 0
    for param in predictor.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Predictor parameters: {total_param}')

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
    
    tb_logger = SummaryWriter(
        os.path.join(
            "logs", 
            f"{args.gnn_type}-L{args.num_layer}-h{args.emb_dim}-d{args.dropout}-LR{args.lr}", 
            time.strftime("%Y%m%dT%H%M%S")
        )
    )
    
    for run in range(args.runs):
        assert args.runs == 1
        # model.reset_parameters()

        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5, verbose=True)

        for epoch in range(1, 1 + args.epochs):

            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, device)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator, args.batch_size, device)

                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    tb_logger.add_scalar('loss', loss, epoch)
                    tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                    
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

                        tb_logger.add_scalar(f'{key}/train_hits', 100 * train_hits, epoch)
                        tb_logger.add_scalar(f'{key}/valid_hits', 100 * valid_hits, epoch)
                        tb_logger.add_scalar(f'{key}/test_hits', 100 * test_hits, epoch)     
                        
                    print('---')
                
                scheduler.step(100 * results["Hits@10"][1])
            
            if optimizer.param_groups[0]['lr'] < 1e-5:
                break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)


if __name__ == "__main__":
    main()