import os
import wandb
import gc
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
import copy
from src.utils.lm_modeling import load_model as load_embed_model, load_text2embedding
from collections import defaultdict
import pandas as pd
import json

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")


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
def add_edges_to_graph(sample, text_attr):
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
        with torch.no_grad():  # 8.5
            text_attr, text_pred, _ = completion_model.inference(sample)
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
        # 如果保存路径被提供，则保存补全后的样本
        if save_dir:
            save_path = os.path.join(save_dir, f"graph_{sample['graph_id']}.json")
            with open(save_path, 'a') as f:
                json.dump(sample, f)
                f.write('\n')
        del text_attr, text_pred, output, temp_sample
        torch.cuda.empty_cache()
        return sample
    else:
        sample['text_pred'] = ''
        del text_attr, text_pred, output, temp_sample
        torch.cuda.empty_cache()
        return sample


# 构造补全后的新的 Dataset 和 DataLoader
def build_updated_dataset(test_loader, completion_model, lp_model, lp_threshold, device, save_dir):
    # 用于按 graph_id 组织所有样本
    graph_sample_groups = defaultdict(list)
    completion_model.eval()
    lp_model.eval()

    with torch.no_grad():
        progress_bar_update = tqdm(range(len(test_loader)))
        for batch in test_loader:
            samples = unpack_batch(batch)
            for sample in samples:
                sample = {k: v.to(device1) if torch.is_tensor(v) else v for k, v in sample.items()}

                updated_sample = process_sample(sample, completion_model, lp_model, lp_threshold, save_dir)

                torch.cuda.empty_cache()
                gc.collect()

                graph_sample_groups[updated_sample['graph_id']].append(updated_sample)

            progress_bar_update.update(1)

    # 生成 embedding 的相关模型
    model_name = 'sbert'
    model, tokenizer, emb_device = load_embed_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    merged_samples = []

    for graph_id, sample_group in graph_sample_groups.items():
        print(f'##graphid:{graph_id}')
        base_sample = sample_group[0]
        graph = copy.deepcopy(base_sample['graph'])
        graph = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in graph.items()}

        # 收集所有新增边和文本描述
        new_edges = []
        pred_texts = []
        src_list = []
        dst_list = []

        for sample in sample_group:
            if sample['text_pred'] != '':
                new_edges.append(torch.tensor([[sample['src']], [sample['dst']]]))  # shape [2, 1]
                pred_texts.append(sample['text_pred'])
                src_list.append(sample['src'].item())
                dst_list.append(sample['dst'].item())

        # 拼接 edge_index
        if new_edges:
            new_edge_index = torch.cat(new_edges, dim=1)
            graph['edge_index'] = torch.cat([graph['edge_index'], new_edge_index], dim=1)

        # 生成 embedding 并拼接 edge_attr
        if pred_texts:
            new_edge_attr = text2embedding(model, tokenizer, emb_device, pred_texts)
            graph['edge_attr'] = torch.cat([graph['edge_attr'], new_edge_attr], dim=0)

        # 复制 question_edge 的第一项来扩展
        if 'question_edge' in graph:
            original_qe = graph['question_edge']  # shape [original_edge_count, D]
            num_new_edges = new_edge_index.size(1) if new_edges else 0

            if num_new_edges > 0:
                first_qe = original_qe[0].unsqueeze(0)  # shape [1, D]
                extended_qe = first_qe.repeat(num_new_edges, 1)  # shape [num_new_edges, D]
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

        from torch_geometric.data import Data
        graph = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in graph.items()}

        graph = Data(**graph)

        merged_sample = {
            'id': graph_id,
            'label': base_sample['label'],
            'graph': graph,
            'desc': desc,
            'question': base_sample['generation_question']
        }
        merged_samples.append(merged_sample)

    class MergedGraphDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __getitem__(self, idx):
            return self.samples[idx]

        def __len__(self):
            return len(self.samples)

    updated_dataset = MergedGraphDataset(merged_samples)

    updated_loader = DataLoader(
        updated_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return updated_dataset, updated_loader


# 调用 LLM 模型生成预测结果
def run_llm_generation(test_loader, llm_model, args):
    llm_model.eval()
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = llm_model.inference(batch)
            eval_output.append(output)

        progress_bar_test.update(1)
    return eval_output


# 主流程入口
def main(args):
    seed = args.seed
    seed_everything(seed=args.seed)
    print(args)

    # Step 2: Build Dataset
    print(f'***start build dataset')
    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    test_dataset = load_dataset[args.dataset](idx_split=idx_split['test'])
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True,
                             shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args).to(device0)
    completion_model = load_model[args.completion_model_name](args=args)
    lp_model = load_model[args.lp_model_name](args=args).to(device0)
    completion_model = _reload_best_model(completion_model, args)
    lp_model = _reload_best_model(lp_model, args).to(device0)
    model = _reload_best_model(model, args).to(device0)

    # 补全并保存数据
    save_dir = '/home/ubuntu/workspace/activeRAG/dataset/webqsp/update'
    build_updated_dataset(test_loader, completion_model, lp_model, lp_threshold=args.lp_threshold, device=device,
                          save_dir=save_dir)

    # 生成阶段：加载补全后的数据并执行 LLM 推理
    updated_samples = []
    for filename in os.listdir(save_dir):
        if filename.endswith('.json'):
            with open(os.path.join(save_dir, filename), 'r') as f:
                updated_samples.append(json.load(f))

    rag_outputs = run_llm_generation(updated_samples, model, args)

    # Step 7. Post-processing & compute metrics
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    acc = eval_funcs[args.dataset](rag_outputs, path)
    print(f'Test Acc {acc}')


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
