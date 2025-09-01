# 文件一：completion_and_lp.py
# 分批次补全，避免 OOM，每个 DataLoader 的 batch 直接处理并释放资源

import os
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import pandas as pd
import copy
from src.dataset import load_dataset
from src.utils.config import parse_args_llama
from src.utils.seed import seed_everything
from src.utils.ckpt import _reload_best_model
from src.utils.collate import collate_fn
from src.model import load_model, llama_model_path
from src.utils.lm_modeling import load_model as load_embed_model, load_text2embedding
from torch_geometric.data import Data

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")

class MergedGraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __getitem__(self, idx):
        return self.samples[idx]
    def __len__(self):
        return len(self.samples)

# 解包 batch，为每个样本构造独立字典
def unpack_batch(batch):
    samples = []
    batch_size = len(batch['graph_id'])
    for i in range(batch_size):
        sample = {
            'graph_id': batch['graph_id'][i],
            'edge_id': batch['edge_id'][i],
            'label': batch['label'][i],
            'graph': batch['graph'][i],
            'desc': batch['desc'][i],
            'question': batch['question'][i],
            'generation_question': batch['generation_question'][i],
            'completion_label': batch['completion_label'][i],
            'lp_label': batch['lp_label'][i],
            'candidate_edge_attr': batch['candidate_edge_attr'][i],
            'candidate_edge_index': batch['candidate_edge_index'][i],
            'src': batch['src'][i],
            'dst': batch['dst'][i],
            'node_df': batch['node_df'][i],
            'edge_df': batch['edge_df'][i]
        }
        samples.append(sample)
    return samples

def update_sample(sample, text_attr):
    edge_index = sample['candidate_edge_index']  # shape [2, E]
    src = sample['src']
    dst = sample['dst']
    src_nodes = edge_index[0]  # shape: [E]
    dst_nodes = edge_index[1]  # shape: [E]

    # 找到 (src, dst) 的匹配位置
    mask = (src_nodes == src) & (dst_nodes == dst)
    indices = torch.nonzero(mask, as_tuple=False).squeeze()

    if indices.numel() == 0:
        raise ValueError(f"(src={src.item()}, dst={dst.item()}) 不在 candidate_edge_index 中")
    elif indices.numel() > 1:
        print(f"Warning: 找到多个 (src, dst)，将全部更新")

    # 赋值 attr（a_attr: [1, D] => [D]）
    sample['candidate_edge_attr'][indices] = text_attr.squeeze(0)
    return sample

# 将新边添加到图结构中
def add_edges_to_graph(sample,text_attr):
    edge_index = sample['candidate_edge_index']  # shape [2, E]
    src = sample['src']
    dst = sample['dst']
    src_nodes = edge_index[0]  # shape: [E]
    dst_nodes = edge_index[1]  # shape: [E]

    mask = (src_nodes == src) & (dst_nodes == dst)
    indices = torch.nonzero(mask, as_tuple=False).squeeze()

    if indices.numel() == 0:
        raise ValueError(f"(src={src.item()}, dst={dst.item()}) 不在 candidate_edge_index 中")
    elif indices.numel() > 1:
        print(f"Warning: 找到多个 (src, dst)，将全部使用第一个")

    idx = indices[0] if indices.ndim > 0 else indices

    # 取得对应边和属性
    new_edge = edge_index[:, idx].unsqueeze(1)  # shape [2, 1]
    new_attr = text_attr  # shape: [1, D] or [D]

    sample['graph']['edge_index'] = torch.cat([sample['graph']['edge_index'], new_edge], dim=1)
    sample['graph']['edge_attr'] = torch.cat([sample['graph']['edge_attr'], new_attr], dim=0)

    return sample

# 8.5
def safe_copy_sample(sample):
    def safe_copy(v):
        if torch.is_tensor(v):
            return v.detach().clone().cpu()
        elif isinstance(v, dict):
            return {kk: safe_copy(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            return [safe_copy(vv) for vv in v]
        else:
            return copy.deepcopy(v)
    return {k: safe_copy(v) for k, v in sample.items()}

# 对单个 sample 执行 completion + link prediction，并返回更新后的 sample。
def process_sample(sample, completion_model, lp_model, lp_threshold, save_dir=None):
    edge_index = sample['graph']['edge_index']  # shape: [2, N]
    src = sample['src'].to(edge_index.device)
    dst = sample['dst'].to(edge_index.device)
    existing_src = edge_index[0]
    existing_dst = edge_index[1]

    is_existing = ((existing_src == src) & (existing_dst == dst)).any()
    if is_existing:
        sample['text_pred'] = ''
        return sample  # 边已存在，跳过处理

    try:
        with torch.no_grad(): # 8.5
            text_attr, text_pred , _ = completion_model.inference(sample)
    except Exception as e:
        print(f"[ERROR] completion_model.inference 崩了: graph_id={sample['graph_id']}, error={e}")
        raise

    temp_sample = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            temp_sample[k] = v.detach().cpu()  # detach to be safe
        else:
            try:
                temp_sample[k] = copy.deepcopy(v)
            except Exception as e:
                print(f"Warning: deepcopy failed for key {k} with error {e}. Skipping...")
                temp_sample[k] = v  # fallback, or use copy.copy(v)

    temp_sample = update_sample(temp_sample, text_attr)
    output = lp_model.inference(temp_sample)

    if output['pred'] > lp_threshold:
        sample['text_pred'] = text_pred[0]
        del text_attr, text_pred, output, temp_sample
        torch.cuda.empty_cache()
        return sample
    else:
        sample['text_pred'] = ''
        del text_attr, text_pred, output, temp_sample
        torch.cuda.empty_cache()
        return sample

# 修改后的流式处理逻辑（不缓存 all_samples）
def build_updated_dataset_streaming(test_loader, completion_model, lp_model, lp_threshold, device, save_dir):
    print("[Info] Streaming processing starts...")

    model_name = 'sbert'
    model, tokenizer, emb_device = load_embed_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    completion_model.eval()
    lp_model.eval()

    with torch.no_grad():
        progress_bar = tqdm(test_loader)
        for batch in progress_bar:
            samples = unpack_batch(batch)
            graph_sample_groups = defaultdict(list)

            for sample in samples:
                sample = {k: v.to(device1) if torch.is_tensor(v) else v for k, v in sample.items()}
                updated_sample = process_sample(sample, completion_model, lp_model, lp_threshold)
                graph_sample_groups[updated_sample['graph_id']].append(updated_sample)

            for graph_id, sample_group in graph_sample_groups.items():
                base_sample = sample_group[0]
                graph = copy.deepcopy(base_sample['graph'])
                graph = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in graph.items()}

                new_edges = []
                pred_texts = []
                src_list = []
                dst_list = []

                for sample in sample_group:
                    if sample['text_pred'] != '':
                        new_edges.append(torch.tensor([[sample['src']], [sample['dst']]]))
                        pred_texts.append(sample['text_pred'])
                        src_list.append(sample['src'].item())
                        dst_list.append(sample['dst'].item())

                if new_edges:
                    new_edge_index = torch.cat(new_edges, dim=1)
                    graph['edge_index'] = torch.cat([graph['edge_index'], new_edge_index], dim=1)

                if pred_texts:
                    new_edge_attr = text2embedding(model, tokenizer, emb_device, pred_texts)
                    graph['edge_attr'] = torch.cat([graph['edge_attr'], new_edge_attr], dim=0)

                if 'question_edge' in graph:
                    original_qe = graph['question_edge']
                    num_new_edges = new_edge_index.size(1) if new_edges else 0
                    if num_new_edges > 0:
                        first_qe = original_qe[0].unsqueeze(0)
                        extended_qe = first_qe.repeat(num_new_edges, 1)
                        graph['question_edge'] = torch.cat([original_qe, extended_qe], dim=0)
                    else:
                        graph['question_edge'] = original_qe

                new_edge_rows = pd.DataFrame({
                    'src': src_list,
                    'edge_attr': pred_texts,
                    'dst': dst_list
                })
                base_sample['edge_df'] = pd.concat([base_sample['edge_df'], new_edge_rows], ignore_index=True)
                node_df = base_sample['node_df']
                desc = node_df.to_csv(index=False) + '\n' + new_edge_rows.to_csv(index=False)

                graph = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in graph.items()}
                graph = Data(**graph)

                merged_sample = {
                    'id': graph_id,
                    'label': base_sample['label'],
                    'graph': graph,
                    'desc': desc,
                    'question': base_sample['generation_question']
                }

                print(f'####merged_data:{merged_sample}')

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{graph_id}.pt")
                    torch.save(merged_sample, save_path)

            del batch, samples, graph_sample_groups
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    args = parse_args_llama()
    seed_everything(seed=args.seed)

    device = torch.device("cuda:0")
    print("[Info] Building dataset...")
    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    test_dataset = load_dataset[args.dataset](idx_split=idx_split['test'])
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    print("[Info] Loading completion & LP model...")
    args.llm_model_path = llama_model_path[args.llm_model_name]
    completion_model = load_model[args.completion_model_name](args=args)
    lp_model = load_model[args.lp_model_name](args=args).to(device)
    completion_model = _reload_best_model(completion_model, args)
    lp_model = _reload_best_model(lp_model, args).to(device)

    print("[Info] Streaming completion & save")
    save_dir = "/home/ubuntu/workspace/activeRAG/dataset/webqsp/tempo_graphs"
    build_updated_dataset_streaming(
        test_loader,
        completion_model,
        lp_model,
        lp_threshold=args.lp_threshold,
        device=device,
        save_dir=save_dir
    )

    torch.cuda.empty_cache()
    gc.collect()
