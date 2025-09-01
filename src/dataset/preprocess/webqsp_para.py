import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding


model_name = 'sbert'
path = 'dataset/webqsp'
# path = '/opt/dlami/nvme/cache/dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

def remove_repeated_edges_tensor(data):
    edge_list = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    edge_dict = {}
    for idx, edge in enumerate(edge_list):
        if edge in edge_dict:
            edge_dict[edge].append(idx)  # Store original indices of duplicates
        else:
            edge_dict[edge] = [idx]  # Start a list with the current index
    
    filtered_indices = [idxs[0] for idxs in edge_dict.values() if len(idxs) == 1]
    data.edge_index = data.edge_index[:, filtered_indices]
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[filtered_indices]

    return data

def step_one():
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)


def generate_split():
    
    dataset = load_dataset("rmanluo/RoG-webqsp")

    train_indices = np.arange(len(dataset['train']))
    val_indices = np.arange(len(dataset['validation'])) + len(dataset['train'])
    test_indices = np.arange(len(dataset['test'])) + len(dataset['train']) + len(dataset['validation'])

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))


def step_two_worker(start_idx, end_idx, device_id):
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    dataset = dataset.select(range(start_idx, end_idx))
    questions = dataset["question"]

    model, tokenizer, device = load_model[model_name](device_id)
    text2embedding = load_text2embedding[model_name]

    print(f"[GPU {device_id}] Encoding graphs from {start_idx} to {end_idx}")
    os.makedirs(path_graphs, exist_ok=True)

    for local_idx, data_point in enumerate(tqdm(dataset)):
        index = start_idx + local_idx

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        nodes.node_attr.fillna("", inplace=True)
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        pyg_graph = remove_repeated_edges_tensor(pyg_graph)

        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')


def step_two_parallel():
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    total_len = len(dataset)

    num_gpus = torch.cuda.device_count()
    chunk = total_len // num_gpus

    processes = []
    for i in range(num_gpus):
        start_idx = i * chunk
        end_idx = (i + 1) * chunk if i < num_gpus - 1 else total_len
        p = mp.Process(target=step_two_worker, args=(start_idx, end_idx, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()




if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    step_one()
    step_two_parallel()
    generate_split()
