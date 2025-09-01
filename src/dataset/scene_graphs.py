import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.utils.graph_retrieval import retrive_on_graphs
from torch_geometric.data.data import Data


model_name = 'sbert'
path = 'dataset/scene_graphs'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'


class SceneGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = None
        self.graph = None
        self.graph_type = 'Scene Graph'
        self.questions = pd.read_csv(f'{path}/questions.csv')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]
        question = f'Question: {data["question"]}\n\nAnswer:'
        graph = torch.load(f'{cached_graph}/{index}.pt', weights_only=False)
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()

        return {
            'id': index,
            'image_id': data['image_id'],
            'question': question,
            'label': data['answer'],
            'full_label': data['full_answer'],
            'graph': graph,
            'desc': desc,
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


def preprocess(topk=3, k = 2, topk_entity = 3, augment = "none"):
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    path_sims = f'{path}/sims_{k}hop_{augment}'
    os.makedirs(path_sims, exist_ok=True)

    questions = pd.read_csv(f'{path}/questions.csv')
    q_embs = torch.load(f'{path}/q_embs.pt', weights_only=False)
    for index in tqdm(range(len(questions))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        image_id = questions.iloc[index]['image_id']
        graph = torch.load(f'{path_graphs}/{image_id}.pt', weights_only=False)
        nodes = pd.read_csv(f'{path_nodes}/{image_id}.csv')
        edges = pd.read_csv(f'{path_edges}/{image_id}.csv')
        q_emb = q_embs[index]

        try:
            sims = torch.load(f'{path_sims}/{index}.pt', weights_only=False)
            subg, desc = retrive_on_graphs(graph, q_emb, nodes, edges, topk=topk, k = k, topk_entity = topk_entity, augment = augment, sims = sims)
        except:
            sims, subgraph = retrive_on_graphs(graph, q_emb, nodes, edges, topk=topk, k = k, topk_entity = topk_entity, augment = augment)
            subg, desc = subgraph
            torch.save(sims, f'{path_sims}/{index}.pt')

        data = Data(x = subg.x, 
                    edge_index = subg.edge_index, 
                    edge_attr = subg.edge_attr, 
                    question_node = q_emb.repeat(subg.x.size(0), 1),
                    question_edge = q_emb.repeat(subg.edge_attr.size(0), 1) if subg.edge_attr is not None else None,
                    num_nodes=subg.num_nodes)
        
        torch.save(data, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':
    preprocess()

    dataset = SceneGraphsDataset()

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
