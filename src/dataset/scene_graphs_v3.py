import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.utils.graph_retrieval import retrive_on_graphs
from torch_geometric.data.data import Data
import random

model_name = 'sbert'
path = 'dataset/scene_graphs'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

NUM_NEG_PER_POS = 0.6


def extract_nodes_edges(desc: str):
    lines = [ln.strip() for ln in desc.strip().split('\n') if ln.strip()]

    subgraphs = []
    current_mode = None
    current_nodes, current_edges = [], []

    def flush():
        nonlocal current_nodes, current_edges
        subgraphs.append({"nodes": current_nodes, "edges": current_edges})
        current_nodes, current_edges = [], []

    for line in lines:
        if line == "node_id,node_attr":
            if current_nodes or current_edges:
                flush()
            current_mode = "node"
        elif line == "src,edge_attr,dst":
            current_mode = "edge"
        else:
            if current_mode == "node":
                current_nodes.append(line)
            elif current_mode == "edge":
                current_edges.append(line)

    if current_nodes or current_edges:
        flush()

    global_nodes = []  # (idx, node_attr)
    global_edges = []  # (src_idx, edge_attr, dst_idx)
    offset = 0

    for sg in subgraphs:
        local = []
        for ln in sg["nodes"]:
            left, right = ln.split(",", 1)
            orig_id = int(left.strip())
            node_attr = right.strip()
            local.append((orig_id, node_attr))

        orig2g = {}
        for local_idx, (orig_id, node_attr) in enumerate(local):
            gidx = offset + local_idx
            orig2g[orig_id] = gidx
            global_nodes.append((gidx, node_attr))

        for ln in sg["edges"]:
            src, edge_attr, dst = ln.split(",", 2)
            src_idx = orig2g[int(src.strip())]
            dst_idx = orig2g[int(dst.strip())]
            global_edges.append((src_idx, edge_attr.strip(), dst_idx))

        offset += len(local)

    node_df = pd.DataFrame(global_nodes, columns=["node_id", "node_attr"])
    edge_df = pd.DataFrame(global_edges, columns=["src", "edge_attr", "dst"])
    return node_df, edge_df


class SceneGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.generation_prompt = 'Please answer the given question.'
        self.completion_prompt = 'Question: Generate a relation about the query from the given src entity to the dst entity. Answer in one word. \n\nAnswer:'
        self.graph = None
        self.graph_type = 'Scene Graph'
        self.questions = pd.read_csv(f'{path}/questions.csv')
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]
        question = f'Question: {data["question"]}\n\nAnswer:'
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()
        nodes, edges = extract_nodes_edges(desc)

        negative_edge_attr = []
        negative_edge_text = []
        negative_labels = []
        negative_edge_index = []
        completion_questions = []
        candidate_edge_text = []
        generation_question = self.generation_prompt

        # -------------------------带mask----------------------------
        # 保留 nodes 和 edges dataframe，后面desc动态生成
        edge_attr = graph.edge_attr
        id_to_attr = dict(zip(nodes['node_id'], nodes['node_attr']))  # node_id -> node_attr

        num_nodes = graph.x.size(0)
        src_ids, dst_ids = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')

        # 只保留 src ≠ dst
        mask = src_ids != dst_ids
        src_ids = src_ids[mask]
        dst_ids = dst_ids[mask]

        candidate_edge_index = torch.stack([src_ids.flatten(), dst_ids.flatten()], dim=0)

        # 现有边集合（正样本）
        existing_edges_set = set((u.item(), v.item()) for u, v in graph.edge_index.t())
        existing_edges_text = {(row.src, row.dst): row.edge_attr for _, row in edges.iterrows()}
        existing_edges_attr = {(u.item(), v.item()): edge_attr[idx] for idx, (u, v) in enumerate(graph.edge_index.t())}

        # 构建正样本
        positive_edge_index = []
        positive_edge_attr = []
        positive_edge_text = []
        positive_labels = []

        existing_edges_list = list(existing_edges_set)
        # 当正样本数量大于 max的时候，只取max，让LLM只学一部分内容
        # max_positive = 200
        # if len(existing_edges_list) > max_positive:
        #     existing_edges_list = random.sample(existing_edges_list, max_positive)

        # for (u, v) in existing_edges_set:
        for (u, v) in existing_edges_list:
            positive_edge_index.append((u, v))
            positive_edge_attr.append(existing_edges_attr[(u, v)])
            positive_edge_text.append(existing_edges_text.get((u, v), ''))
            positive_labels.append(1)

        # 构建负样本
        all_edges_set = set((u.item(), v.item()) for u, v in candidate_edge_index.t())
        negative_edges_set = list(all_edges_set - existing_edges_set)

        # 每个正样本配n个负样本
        num_negative_per_positive = NUM_NEG_PER_POS
        num_negative_samples = int(num_negative_per_positive * len(positive_edge_index))
        # num_negative_samples = len(positive_edge_index) // 2

        if len(negative_edges_set) > num_negative_samples:
            negative_edges_set = random.sample(negative_edges_set, num_negative_samples)

        for (u, v) in negative_edges_set:
            negative_edge_index.append((u, v))
            negative_edge_attr.append(torch.zeros(edge_attr.size(1)))  # 1024维全0向量
            negative_edge_text.append('')  # 空文本
            negative_labels.append(0.1)

        # 合并正负样本
        all_edge_index = positive_edge_index + negative_edge_index
        all_edge_attr = positive_edge_attr + negative_edge_attr
        all_edge_text = positive_edge_text + negative_edge_text
        all_labels = positive_labels + negative_labels

        # 转 Tensor
        candidate_edge_index = torch.tensor(all_edge_index, dtype=torch.long).T  # [2, num_edges]
        candidate_edge_attr = torch.stack(all_edge_attr, dim=0)
        edge_label = torch.tensor(all_labels, dtype=torch.float32)  # [num_edges]
        candidate_edge_text = all_edge_text

        # 生成 completion questions，负样本也有
        for src, dst in zip(candidate_edge_index[0], candidate_edge_index[1]):
            src_attr = id_to_attr.get(src.item())
            dst_attr = id_to_attr.get(dst.item())
            completion_question = f"Given two entities:\nSrc entity 1: {src_attr}\nDst entity 2: {dst_attr}\n{self.completion_prompt}"
            completion_questions.append(completion_question)

        # -------------------------带mask----------------------------

        return {
            'id': index,
            'label': label,
            'nodes_df': nodes,
            'edges_df': edges,
            'graph': graph,
            'completion_question': completion_questions,  # edge_completion部分的prompt，是一个list
            'generation_question': generation_question,  # 最终generation部分的prompt
            # 'mask_index': mask,
            'candidate_edge_index': candidate_edge_index,  # [2, num_candidate_edges] 所有节点对的edge_index
            'candidate_edge_attr': candidate_edge_attr,  # [num_candidate_edges, 1024] 所有节点对的edge_attr
            'lp_edge_label': edge_label,  # [num_candidate_edges] 进行link prediction的label
            'completion_edge_label': candidate_edge_text,  # list, [num_candidate_edges] 进行edge completion的text label
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


class EdgePairDataset(Dataset):
    def __init__(self, idx_split=None):
        super().__init__()
        self.base_dataset = SceneGraphsDataset()
        self.samples = []
        self.idx_split = idx_split if idx_split else self.base_dataset.get_idx_split()

        if idx_split:
            for idx in self.idx_split:
                item = self.base_dataset[idx]
                if item['candidate_edge_index'].shape[0] != 0:
                    for edge_idx in range(item['candidate_edge_index'].shape[1]):
                        self.samples.append((idx, edge_idx))
                else:
                    self.samples.append((-1, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        graph_idx, edge_idx = self.samples[index]
        item = self.base_dataset[graph_idx]

        src = item['candidate_edge_index'][0, edge_idx]
        dst = item['candidate_edge_index'][1, edge_idx]

        nodes_df = item['nodes_df']
        edges_df = item['edges_df']

        edges_df['src'] = edges_df['src'].astype(int)
        edges_df['dst'] = edges_df['dst'].astype(int)

        lp_label = item['lp_edge_label'][edge_idx]
        if lp_label == 1:  # 正样本，删掉
            mask = ~((edges_df['src'] == int(src)) & (edges_df['dst'] == int(dst)))
            modified_edges_df = edges_df[mask]
            # modified_edges_df = edges_df
        else:
            modified_edges_df = edges_df  # 负样本，不删

        # 重新生成 desc
        desc = nodes_df.to_csv(index=False) + '\n' + modified_edges_df.to_csv(index=False)

        sample = {
            'graph_id': graph_idx,
            'edge_id': edge_idx,
            'label': item['label'],
            'graph': item['graph'],
            'desc': desc,
            'question': item['completion_question'][edge_idx],
            'generation_question': item['generation_question'],
            'completion_label': item['completion_edge_label'][edge_idx],
            'lp_label': item['lp_edge_label'][edge_idx],
            'candidate_edge_attr': item['candidate_edge_attr'],
            'candidate_edge_index': item['candidate_edge_index'],
            'src': src,
            'dst': dst,
            'node_df': nodes_df,
            'edge_df': modified_edges_df,
        }

        return sample

    def get_idx_split(self):
        return self.idx_split


def preprocess(topk=3, k=2, topk_entity=3, augment="none"):
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    path_sims = f'{path}/sims_{k}hop_{augment}'
    os.makedirs(path_sims, exist_ok=True)

    questions = pd.read_csv(f'{path}/questions.csv')
    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(questions))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        image_id = questions.iloc[index]['image_id']
        graph = torch.load(f'{path_graphs}/{image_id}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{image_id}.csv')
        edges = pd.read_csv(f'{path_edges}/{image_id}.csv')
        q_emb = q_embs[index]

        try:
            sims = torch.load(f'{path_sims}/{index}.pt')
            subg, desc = retrive_on_graphs(graph, q_emb, nodes, edges, topk=topk, k=k, topk_entity=topk_entity,
                                           augment=augment, sims=sims)
        except:
            sims, subgraph = retrive_on_graphs(graph, q_emb, nodes, edges, topk=topk, k=k, topk_entity=topk_entity,
                                               augment=augment)
            subg, desc = subgraph
            torch.save(sims, f'{path_sims}/{index}.pt')

        data = Data(x=subg.x,
                    edge_index=subg.edge_index,
                    edge_attr=subg.edge_attr,
                    question_node=q_emb.repeat(subg.x.size(0), 1),
                    question_edge=q_emb.repeat(subg.edge_attr.size(0), 1) if subg.edge_attr is not None else None,
                    num_nodes=subg.num_nodes)

        torch.save(data, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':
    preprocess()

    dataset = SceneGraphsDataset()

    # data = dataset[0]
    # for k, v in data.items():
    #     print(f'{k}: {v}')

    # split_ids = dataset.get_idx_split()
    # for k, v in split_ids.items():
    #     print(f'# {k}: {len(v)}')
