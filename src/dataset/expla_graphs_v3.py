import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

PATH = 'dataset/expla_graphs'


class ExplaGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.text = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
        # generation_prompt是为了最终generation部分的prompt，completion_prompt是为了edge_completion中，生成text_attribute的prompt
        # self.generation_prompt = 'Do argument 1 and argument 2 support or counter each other? Reply ONLY with one word: \'support\' or \'counter\'.'
        self.generation_prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.completion_prompt = 'Question: Generate a relation about the query from the given src entity to the dst entity. Answer in one word. \n\nAnswer:'
        self.graph = None
        self.graph_type = 'Explanation Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):

        # -------------------------带mask----------------------------
        text = self.text.iloc[index]
        graph = torch.load(f'{PATH}/graphs/{index}.pt', weights_only=False)
        nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
        edges = pd.read_csv(f'{PATH}/edges/{index}.csv')

        # 保留 nodes 和 edges dataframe，后面desc动态生成
        edge_attr = graph.edge_attr
        id_to_attr = dict(zip(nodes['node_id'], nodes['node_attr']))  # node_id -> node_attr

        generation_question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{self.generation_prompt}'

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

        for (u, v) in existing_edges_set:
            positive_edge_index.append((u, v))
            positive_edge_attr.append(existing_edges_attr[(u, v)])
            positive_edge_text.append(existing_edges_text.get((u, v), ''))
            positive_labels.append(1)

        # 构建负样本
        all_edges_set = set((u.item(), v.item()) for u, v in candidate_edge_index.t())
        negative_edges_set = list(all_edges_set - existing_edges_set)

        # 每个正样本配n个负样本
        num_negative_per_positive = 1
        num_negative_samples = int(num_negative_per_positive * len(positive_edge_index))
        # num_negative_samples = len(positive_edge_index) // 2

        if len(negative_edges_set) > num_negative_samples:
            negative_edges_set = random.sample(negative_edges_set, num_negative_samples)

        negative_edge_attr = []
        negative_edge_text = []
        negative_labels = []
        negative_edge_index = []

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
        candidate_edge_attr = torch.stack(all_edge_attr, dim=0)  # [num_edges, 1024]
        edge_label = torch.tensor(all_labels, dtype=torch.long)  # [num_edges]
        candidate_edge_text = all_edge_text

        # 生成 completion questions，负样本也有
        completion_questions = []
        for src, dst in zip(candidate_edge_index[0], candidate_edge_index[1]):
            src_attr = id_to_attr.get(src.item())
            dst_attr = id_to_attr.get(dst.item())
            completion_question = f"Given two entities:\nSrc entity 1: {src_attr}\nDst entity 2: {dst_attr}\n{self.completion_prompt}"
            completion_questions.append(completion_question)
        # -------------------------带mask----------------------------

        return {
            'id': index,
            'label': text['label'],
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
        with open(f'{PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


class EdgePairDataset(Dataset):
    def __init__(self, idx_split=None):
        super().__init__()
        self.base_dataset = ExplaGraphsDataset()
        self.samples = []
        self.idx_split = idx_split if idx_split else self.base_dataset.get_idx_split()

        if idx_split:
            for idx in self.idx_split:
                item = self.base_dataset[idx]
                for edge_idx in range(item['candidate_edge_index'].shape[1]):
                    self.samples.append((idx, edge_idx))

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


if __name__ == '__main__':
    dataset = ExplaGraphsDataset()

