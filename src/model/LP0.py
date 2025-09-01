import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.data import Data
from src.model.gnn import load_gnn_model

device = "cuda" if torch.cuda.is_available() else "cpu"

class GNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add') 
        self.message_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.phi = nn.Sequential(  
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.update_fn = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, q_emb):
        # q_expand = q_emb.repeat(edge_attr.size(0),1)
        qr_mul = q_emb * edge_attr  # [num_edges, hidden_dim], 想一下是不是要先求和然后直接sigmoid
        edge_weight = self.phi(qr_mul).squeeze(-1)  # [num_edges]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * self.message_mlp(x_j)

    def update(self, aggr_out, x):
        return self.update_fn(aggr_out, x)

class LinkPredictionModel(nn.Module):
    def __init__(self, args, **kargs):
        super().__init__()
        hidden_dim = args.gnn_hidden_dim
        input_dim = args.gnn_in_dim
        num_layers = 2
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim + hidden_dim, hidden_dim), #后面看一下有没有对齐
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, samples):
        #----------------------------batch版本-------------------------
        graph = samples['graph']  # PyG Batch 对象
        x = graph.x                          # [total_nodes, hidden_dim]
        edge_index = graph.edge_index       # [2, total_edges]
        edge_attr = graph.edge_attr         # [total_edges, hidden_dim]
        
        candidate_edge_index = samples['candidate_edge_index']
        candidate_edge_attr = samples['candidate_edge_attr']

        batch_size = len(samples['graph_id'])
        q_emb_list = []
        q_expand_list = []
        c_edge_index_list = []
        c_edge_attr_list = []
        src_list = []
        dst_list = []
        node_offset = 0
        for i in range (batch_size):
            g = graph[i]
            q_emb = g.question_node[:1]
            q_expand = q_emb.repeat(candidate_edge_attr[i].size(0), 1)
            q_expand_list.append(q_expand)
            q_emb_list.append(q_emb)
            c_edge_idx = candidate_edge_index[i] + node_offset  # [2, E_i] + offset
            c_edge_attr = candidate_edge_attr[i]                 # [E_i, D]
            c_edge_index_list.append(c_edge_idx)
            c_edge_attr_list.append(c_edge_attr)
            src = samples['src'][i] + node_offset
            dst = samples['dst'][i] + node_offset
            src_list.append(src)
            dst_list.append(dst)
            node_offset += g.num_nodes

        q_expand = torch.cat(q_expand_list, dim=0).to(x.device)
        q_emb = torch.cat(q_emb_list, dim=0).to(x.device)
            
        candidate_edge_index = torch.cat(c_edge_index_list, dim=1).to(x.device)  # [2, total_E]
        candidate_edge_attr = torch.cat(c_edge_attr_list, dim=0).to(x.device) 
        src_list = torch.stack(src_list, dim=0).long().to(x.device)
        dst_list = torch.stack(dst_list, dim=0).long().to(x.device)

        # 输入投影
        x = self.input_proj(x)

        # 图神经网络层
        for gnn in self.gnn_layers:
            x = gnn(x, candidate_edge_index, candidate_edge_attr, q_expand)  # q_emb 应该是 [B, D]，PyG 自定义层应支持 batch

        # 将 src, dst list 变成张量
        src_list = torch.stack(samples['src'], dim=0).long().to(x.device)  # [B]
        dst_list = torch.stack(samples['dst'], dim=0).long().to(x.device)  # [B]
        labels = torch.stack(samples['lp_label'], dim=0).float().to(x.device)  # [B]

        h_src = x[src_list]   # [B, hidden_dim]
        h_dst = x[dst_list]   # [B, hidden_dim]

        # 如果 q_emb 是 [B, hidden_dim]，不需要 repeat
        edge_feat = torch.cat([h_src, h_dst, h_src * h_dst, q_emb ], dim=-1)  # [B, 4*hidden_dim]

        logits = self.edge_mlp(edge_feat).squeeze(-1)  # [B]
        loss = self.loss_fn(logits, labels)

        # pos_mask = (labels == 1)
        # unlabeled_mask = (labels == 0)

        # pos_logits = logits[pos_mask]
        # unlabeled_logits = logits[unlabeled_mask]

        # pi_p = 0.2

        # pos_loss = self.bce_loss(pos_logits, torch.ones_like(pos_logits))
        # pos_loss_neg = self.bce_loss(pos_logits, torch.zeros_like(pos_logits))
        # unlabeled_loss = self.bce_loss(unlabeled_logits, torch.zeros_like(unlabeled_logits))

        # positive_risk = pi_p * pos_loss.mean()
        # negative_risk = unlabeled_loss.mean() - pi_p * pos_loss_neg.mean()

        # loss = positive_risk + torch.clamp(negative_risk, min=0.0)

        #----------------------------batch版本-------------------------

        #----------------------------非batch版本-------------------------
        # graph = samples['graph']
        # x = graph.x
        # edge_index = graph.edge_index
        # edge_attr = graph.edge_attr # [num_edges, hidden_dim]
        # q_emb = graph.question_node[:1] #[1, hidden_dim] 
        # label = samples['lp_label'].to(device)
         
        # candidate_edge_index = samples['candidate_edge_index'].to(x.device)
        # candidate_edge_attr = samples['candidate_edge_attr'].to(x.device)

        # x = self.input_proj(x)  # [num_nodes, hidden_dim]

        # for gnn in self.gnn_layers:
        #     x = gnn(x, candidate_edge_index, candidate_edge_attr, q_emb)
        # src = samples['src']
        # dst = samples['dst']
        # h_src = x[src]
        # h_dst = x[dst]
        # # q_expand = q_emb.repeat(h_src.size(0), 1)
        # q_emb = q_emb.squeeze(0)
        # edge_feat = torch.cat([
        #     h_src, h_dst, h_src * h_dst, q_emb
        # ], dim=-1)
        # logits = self.edge_mlp(edge_feat).squeeze(-1)  
        # # probs = torch.sigmoid(logits) 
        # # pred = (probs > 0.5)
        # loss = self.loss_fn(logits, label.float())
        #----------------------------非batch版本-------------------------

        return loss 
    
    def inference(self, samples):
        graph = samples['graph']
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr # [num_edges, hidden_dim]
        q_emb = graph.question_node[:1] #[1, hidden_dim] 

        candidate_edge_index = samples['candidate_edge_index'].to(x.device)
        candidate_edge_attr = samples['candidate_edge_attr'].to(x.device)  

        x = self.input_proj(x)  # [num_nodes, hidden_dim]

        for gnn in self.gnn_layers:
            # x = gnn(x, edge_index, edge_attr, q_emb)
            x = gnn(x, candidate_edge_index, candidate_edge_attr, q_emb)

        src = samples['src']
        dst = samples['dst']

        h_src = x[src]
        h_dst = x[dst]
        # q_expand = q_emb.repeat(h_src.size(0), 1)
        q_emb = q_emb.squeeze(0)

        edge_feat = torch.cat([
            h_src, h_dst, h_src * h_dst, q_emb
        ], dim=-1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)  
        probs = torch.sigmoid(logits) 
        # print(f'probs:{probs}')
        pred = (probs > 0.5)

        return {'graph_id': samples['graph_id'],
                'edge_id': samples['edge_id'],
                'pred': pred, }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param