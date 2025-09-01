import os
import json
import re
from typing import Optional, Tuple
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data.data import Data
from src.utils.generate_split import generate_split
from src.utils.lm_modeling import load_model, load_text2embedding

PATH = "/home/ubuntu/workspace/gfm-rag-main/data/expla_graph_small/raw"
KG_PATH = "/home/ubuntu/workspace/gfm-rag-main/data/expla_graph_small/processed/stage1/kg copy.txt"
TEST_JSON = "/home/ubuntu/workspace/gfm-rag-main/data/expla_graph_small/raw/test.json"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'sbert'

def _parse_triplet_line(line: str) -> Optional[Tuple[str, str, str]]:
    if not line.strip():
        return None
    parts = line.rstrip("\n").split(",", 2)
    if len(parts) != 3:
        parts = re.split(r"[,\t]", line.rstrip("\n"))
        if len(parts) < 3:
            return None
        h, r, t = parts[0], parts[1], ",".join(parts[2:])
    else:
        h, r, t = parts
    return h.strip().lower(), r.strip().lower(), t.strip().lower()

def build_from_kg():
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    nodes_dir = os.path.join(PATH, "nodes")
    edges_dir = os.path.join(PATH, "edges")
    graphs_dir = os.path.join(PATH, "graphs")

    os.makedirs(nodes_dir, exist_ok=True)
    os.makedirs(edges_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = [item.get("question", "").strip() for item in dataset]
    num_graphs = len(questions)
    print(f"[Step 0] Loaded {len(questions)} question from {TEST_JSON}")

    print("[Step 1] Parsing KG and writing nodes/edges CSVs...")
    node2id = {}
    edges = []
    with open(KG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            trip = _parse_triplet_line(line)
            if trip is None:
                continue
            h, r, t = trip
            if h not in node2id:
                node2id[h] = len(node2id)
            if t not in node2id:
                node2id[t] = len(node2id)
            edges.append({"src": node2id[h], "edge_attr": r, "dst": node2id[t]})

    nodes_df = pd.DataFrame(
        [{"node_id": nid, "node_attr": n} for n, nid in node2id.items()],
        columns=["node_id", "node_attr"],
    ).sort_values("node_id")
    edges_df = pd.DataFrame(edges, columns=["src", "edge_attr", "dst"])
    nodes_df.to_csv(os.path.join(nodes_dir, "0.csv"), index=False, columns=["node_id", "node_attr"])
    edges_df.to_csv(os.path.join(edges_dir, "0.csv"), index=False, columns=["src", "edge_attr", "dst"])

    print("[Step 2] Encoding questions -> q_embs.pt ...")
    q_embs = text2embedding(model, tokenizer, device, questions)
    q_embs_path = os.path.join(PATH, "q_embs.pt")
    torch.save(q_embs, q_embs_path)

    print("[Step 3] Encoding graphs and saving Data objects ...")
    i = 0
    node = pd.read_csv(os.path.join(nodes_dir, f"{i}.csv"))
    edge = pd.read_csv(os.path.join(edges_dir, f"{i}.csv"))

    x = text2embedding(model, tokenizer, device, node["node_attr"].astype(str).tolist())
    e = text2embedding(model, tokenizer, device, edge["edge_attr"].astype(str).tolist())

    src = torch.as_tensor(edge["src"].values, dtype=torch.long)
    dst = torch.as_tensor(edge["dst"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)                # [2, E]

    qi = q_embs[i]                                             # [D]
    question_node = qi.unsqueeze(0).repeat(x.size(0), 1)       # [N, D]
    question_edge = qi.unsqueeze(0).repeat(e.size(0), 1)       # [E, D]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=e,
        question_node=question_node,
        question_edge=question_edge,
        num_nodes=len(node),
    )
    torch.save(data, os.path.join(graphs_dir, f"{i}.pt"))


def get_dataset():

    text = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
    generation_prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
    completion_prompt = 'Question: Generate a relation about the query from the given src entity to the dst entity. Answer in one word. \n\nAnswer:'

    text = text.iloc[index]
    graph = torch.load(f'{PATH}/graphs/{index}.pt', weights_only=False)
    nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
    edges = pd.read_csv(f'{PATH}/edges/{index}.csv')

    # 保留 nodes 和 edges dataframe，后面desc动态生成
    edge_attr = graph.edge_attr
    id_to_attr = dict(zip(nodes['node_id'], nodes['node_attr']))  # node_id -> node_attr

    generation_question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{generation_prompt}'

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
    num_negative_samples = num_negative_per_positive * len(positive_edge_index)

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
    candidate_edge_attr = torch.stack(all_edge_attr, dim=0)                 # [num_edges, 1024]
    edge_label = torch.tensor(all_labels, dtype=torch.long)                 # [num_edges]
    candidate_edge_text = all_edge_text

    # 生成 completion questions，负样本也有
    completion_questions = []
    for src, dst in zip(candidate_edge_index[0], candidate_edge_index[1]):
        src_attr = id_to_attr.get(src.item())
        dst_attr = id_to_attr.get(dst.item())
        completion_question = f"Given two entities:\nSrc entity 1: {src_attr}\nDst entity 2: {dst_attr}\n{completion_prompt}"
        completion_questions.append(completion_question)


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
        'label': text['label'],
        'graph': graph,
        'desc': desc,
        'question': completion_question[edge_idx],
        'generation_question': generation_question,
        'completion_label': candidate_edge_text[edge_idx],
        'lp_label': edge_label[edge_idx],
        'candidate_edge_attr': 'candidate_edge_attr',
        'candidate_edge_index':  'candidate_edge_index',
        'src': src,
        'dst': dst,
        'node_df': nodes_df,
        'edge_df': modified_edges_df,
    }

    return sample



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
            # print(f"text:{text_pred}")
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

    # print(f'pred:{output['pred']}, threshold:{lp_threshold}')

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


                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{graph_id}.pt")
                    torch.save(merged_sample, save_path)

            del batch, samples, graph_sample_groups
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":

    build_from_kg()
    dataset = get_dataset()

    args = parse_args_llama()
    seed_everything(seed=args.seed)

    device = torch.device("cuda:0")
    print("[Info] Building dataset...")
    test_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    print("[Info] Loading completion & LP model...")
    args.llm_model_path = llama_model_path[args.llm_model_name]
    completion_model = load_model[args.completion_model_name](args=args)
    lp_model = load_model[args.lp_model_name](args=args).to(device)
    completion_model = _reload_best_model(completion_model, args)
    lp_model = _reload_best_model(lp_model, args).to(device)

    print("[Info] Streaming completion & save")
    save_dir = "/home/ubuntu/workspace/activeRAG/dataset/scene_graphs/tempo_graphs"
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


