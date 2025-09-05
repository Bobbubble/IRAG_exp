import os
import wandb
import gc
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

graph_cache = defaultdict(lambda: {
    'completion_loss': [],
    'edge_attr_preds': [],
    'edge_attr_labels': [],
    'edge_index': None,
    'graph_data': None,
})


def unpack_batch(batch):
    samples = []
    batch_size = len(batch['graph_id'])
    for i in range(batch_size):
        sample = {
            'graph_id': batch['graph_id'][i],
            'edge_id': batch['edge_id'][i],
            'graph': batch['graph'][i],
            'desc': batch['desc'][i],
            'question': batch['question'][i],
            'completion_label': batch['completion_label'][i],
            'lp_label': batch['lp_label'][i],
            'candidate_edge_attr': batch['candidate_edge_attr'][i],
            'candidate_edge_index': batch['candidate_edge_index'][i],
            'src': batch['src'][i],
            'dst': batch['dst'][i],
        }
        samples.append(sample)
    return samples


def update_sample(sample, text_attr, mode='train'):
    # ----------------非batch版本--------------------------------------
    if mode == 'test':
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
        # sample['lp_label'][indices] = 1
    # ----------------非batch版本--------------------------------------

    # ----------------batch版本--------------------------------------
    elif mode == 'train':
        batch_size = len(sample['graph_id'])
        for i in range(batch_size):
            edge_index = sample['candidate_edge_index'][i]  # shape [2, E]
            src = sample['src'][i]
            dst = sample['dst'][i]
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            # 找到 (src, dst) 的匹配边
            mask = (src_nodes == src) & (dst_nodes == dst)
            indices = torch.nonzero(mask, as_tuple=False).squeeze()

            if indices.numel() == 0:
                raise ValueError(f"(src={src.item()}, dst={dst.item()}) 不在 candidate_edge_index 中")

            # 更新属性
            sample['candidate_edge_attr'][i][indices] = text_attr[i]
    # ----------------batch版本--------------------------------------

    return sample


def main(args):
    # Step 1: Set up wandb
    seed = args.seed
    # wandb.init(project=f"{args.project}",
    #            name=f"{args.dataset}_{args.model_name}_seed{seed}",
    #            config=args)

    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Dataset
    print(f'***start build dataset')
    train_dataset = load_dataset[args.dataset](idx_split=idx_split['train'])
    print(f'***train_dataset')
    val_dataset = load_dataset[args.dataset](idx_split=idx_split['val'])
    print(f'***val_dataset')
    test_dataset = load_dataset[args.dataset](idx_split=idx_split['test'])
    print(f'***test_dataset')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True,
                             shuffle=False, collate_fn=collate_fn)

    print(f'***finish loader')

    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    completion_model = load_model[args.completion_model_name](args=args).to(device1)
    lp_model = load_model[args.lp_model_name](args=args).to(device0)

    # Step 4 Set Optimizer
    params = list(p for p in completion_model.parameters() if p.requires_grad) + \
             list(p for p in lp_model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = completion_model.print_trainable_params()
    print(
        f"completion trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    trainable_params, all_param = lp_model.print_trainable_params()
    print(
        f"lp trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):

        completion_model.train()
        completion_epoch_loss, completion_accum_loss = 0., 0.
        lp_model.train()
        lp_epoch_loss, lp_accum_loss = 0., 0.
        total_epoch_loss, total_accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_loss = 0.0

            # ----------------batch版本--------------------------------------
            total_lp_loss = 0.0
            completion_loss, pred_attr = completion_model(batch, return_attr=True)
            batch = update_sample(batch, pred_attr, 'train')
            lp_loss = lp_model(batch).to(device1)
            total_loss = completion_loss + lp_loss
            batch_loss += total_loss
            # ----------------batch版本--------------------------------------

            # ----------------非batch版本--------------------------------------
            # samples = unpack_batch(batch)
            # # print(f'samples:{samples}')
            # for sample in samples:  # 每个sample是一个edge对
            #     # print(f'sample:{sample}')
            #     graph_id = sample['graph_id']
            #     edge_id = sample['edge_id']
            #     completion_loss, pred_attr = completion_model(sample, return_attr=True)
            #     sample = update_sample(sample,pred_attr)
            #     lp_loss = lp_model(sample)
            #     total_loss = completion_loss + lp_loss
            #     batch_loss += total_loss
            # ----------------非batch版本--------------------------------------

            batch_loss.backward()
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            completion_epoch_loss, completion_accum_loss = completion_epoch_loss + completion_loss.item(), completion_accum_loss + completion_loss.item()
            lp_epoch_loss, lp_accum_loss = lp_epoch_loss + lp_loss.item(), lp_accum_loss + lp_loss.item()
            total_epoch_loss, total_accum_loss = total_epoch_loss + total_loss.item(), total_accum_loss + total_loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                # wandb.log({'Lr': lr})
                # wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                completion_accum_loss = 0.
                lp_accum_loss = 0.
                total_accum_loss = 0.

            progress_bar.update(1)

        print(
            f"Epoch: {epoch}|{args.num_epochs}: Completion Train Loss (Epoch Mean): {completion_epoch_loss / len(train_loader)}")
        # wandb.log({'Completion Train Loss (Epoch Mean)': completion_epoch_loss / len(train_loader)})
        print(f"Epoch: {epoch}|{args.num_epochs}: Lp Train Loss (Epoch Mean): {lp_epoch_loss / len(train_loader)}")
        # wandb.log({'Lp Train Loss (Epoch Mean)': lp_epoch_loss / len(train_loader)})
        print(
            f"Epoch: {epoch}|{args.num_epochs}: Total Train Loss (Epoch Mean): {total_epoch_loss / len(train_loader)}")
        # wandb.log({'Total Train Loss (Epoch Mean)': total_epoch_loss / len(train_loader)})

        val_loss = 0.
        eval_output = []
        completion_model.eval()
        lp_model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                # ----------------batch版本--------------------------------------
                completion_loss, pred_attr = completion_model(batch, return_attr=True)
                batch = update_sample(batch, pred_attr, 'train')
                lp_loss = lp_model(batch).to(device1)
                total_loss = completion_loss + lp_loss
                val_loss += total_loss.item()
                # ----------------batch版本--------------------------------------

                # ----------------非batch版本--------------------------------------
                # samples = unpack_batch(batch)
                # for sample in samples:
                #     completion_loss, pred_attr = completion_model(sample, return_attr=True)
                #     sample = update_sample(sample,pred_attr)
                #     lp_loss = lp_model(sample)
                #     total_loss = completion_loss + lp_loss
                #     val_loss += total_loss.item()
                # ----------------非batch版本--------------------------------------

            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Completion Val Loss: {completion_loss}")
            # wandb.log({'Val Loss': val_loss})
            print(f"Epoch: {epoch}|{args.num_epochs}: Lp Val Loss: {lp_loss}")
            # wandb.log({'Val Loss': val_loss})
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            # wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(completion_model, optimizer, epoch, args, is_best=True)
            _save_checkpoint(lp_model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # ------------------------------后面还要接入generate model-----------------------------------
    # Step 5. Evaluating
    completion_model = _reload_best_model(completion_model, args)
    completion_model.eval()
    lp_model = _reload_best_model(lp_model, args)
    lp_model.eval()
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))

    total_correct = 0  # 累计correct数量
    total_samples = 0  # 总样本数
    total_lp = 0

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            samples = unpack_batch(batch)
            for sample in samples:
                pred_attr, pred_text, correct = completion_model.inference(sample)

                total_correct += correct
                total_samples += 1

                sample = update_sample(sample, pred_attr, 'test')
                output = lp_model.inference(sample)
                eval_output.append(output)

                total_lp += output['pred']

            progress_bar_test.update(1)

    correct_rate = total_correct / total_samples if total_samples > 0 else 0
    lp_rate = total_lp / total_samples if total_samples > 0 else 0
    print(f"\nCorrect rate: {correct_rate:.4f}")
    print(f"\nLP rate: {lp_rate:.4f}")

    # Step 6. Post-processing & compute metrics
    # os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    # path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    # acc = eval_funcs[args.dataset](eval_output, path)
    # print(f'Test Acc {acc}')
    # wandb.log({'Test Acc': acc})


if __name__ == "__main__":
    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
