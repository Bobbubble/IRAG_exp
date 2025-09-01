import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import os
import torch.nn.functional as F
import gc

# 这里是completion model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100

class CompletionModel(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            # "max_memory": {0: '22GiB',1: '22GiB',2: '22GiB',3: '22GiB'},
            "max_memory": {1: '22GiB',2: '22GiB',3: '22GiB'},
            # "max_memory": {2: '22GiB',3: '22GiB'},
            "device_map": "auto",
            # "device_map" : {"": 1},
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=True, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        # self.tokenizer.pad_token = self.tokenizer.eos_token # llama3
        self.tokenizer.padding_side = 'left'

        global EOS
        EOS = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for _, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            mlp_layers = args.alignment_mlp_layers,
            num_heads=args.gnn_num_heads,
            operator=args.distance_operator,
        ).to(self.model.device)

        # If you are using llama2-13b, replace with nn.Linear(2048, 5120) ...
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
            # nn.Linear(2048, 5120),
        ).to(self.model.device)

        self.lp_projector = nn.Sequential(
            nn.Linear(4096, 2048),
            # nn.Linear(5120, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, args.gnn_hidden_dim),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            # return torch.cuda.amp.autocast(dtype=dtype)
            return torch.amp.autocast('cuda',dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, sample, mode='train'):
        graph = sample['graph']
        graph = graph.to(self.model.device)
        n_embed, _ = self.graph_encoder(graph.x, 
                                         graph.edge_index.long(), 
                                         graph.question_node,
                                         graph.edge_attr, 
                                         graph.question_edge)

        # mean pooling
        #--------------------------batch----------------------------
        if mode=='train' or mode =='vali':
            g_embed = scatter(n_embed, graph.batch, dim=0, reduce='mean') 
        #--------------------------batch---------------------------

        #--------------------------非batch---------------------------
        elif mode == 'test':
            g_embed = n_embed.mean(dim=0, keepdim=True) 
        #--------------------------非batch---------------------------

        return g_embed

    def forward(self, samples, return_attr=False):
        """
        训练阶段: 基于query+graph，预测每一条candidate_edge的text attribute
        """
        #--------------------------------batch版本---------------------------------------
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["completion_label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples, 'train')
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['graph_id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            if len(graph_embeds)!=batch_size and i>=batch_size-1: break
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            if len(graph_embeds)!=batch_size and i>=batch_size-1: break
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels= label_input_ids,
                output_hidden_states=True
            )
            # generated_ids = self.model.generate(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=attention_mask,
            #     max_new_tokens=self.max_new_tokens,
            #     use_cache=True
            # )
        # text_attr = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # print(f'text_attr:{text_attr}')
        
        #--------------------------------batch版本---------------------------------------
        
        #--------------------------------非batch版本---------------------------------------
        # label = sample["completion_label"]
        # question = sample['question']
        # description = self.tokenizer(sample["desc"], add_special_tokens=False)

        # graph_embeds = self.encode_graphs(sample)
        # graph_embeds = self.projector(graph_embeds)

        # eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        # eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        # bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))

        # question_tokens = self.tokenizer(question, add_special_tokens=False)

        # label_tokens = self.tokenizer(label, add_special_tokens=False)
        # label_input_ids = label_tokens.input_ids + eos_tokens.input_ids
        # input_ids = description.input_ids + question_tokens.input_ids + eos_user_tokens.input_ids + label_input_ids
        # inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
        # inputs_embeds = torch.cat([bos_embeds, graph_embeds, inputs_embeds], dim=0) # 后面看看这里维度拼接怎么样

        # label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
        # attention_mask = [1] * inputs_embeds.shape[0]

        # attention_mask = torch.tensor(attention_mask).to(self.model.device)
        # label_input_ids = torch.tensor(label_input_ids).to(self.model.device)
       

        # # print(f"inputs_embeds shape: {inputs_embeds.unsqueeze(0).shape}")
        # # print(f"attention_mask shape: {attention_mask.unsqueeze(0).shape}")
        # # print(f"label_input_ids shape: { label_input_ids.unsqueeze(0).shape}")

        # with self.maybe_autocast():
        #     outputs = self.model(
        #         inputs_embeds=inputs_embeds.unsqueeze(0),
        #         attention_mask=attention_mask.unsqueeze(0),
        #         return_dict=True,
        #         labels= label_input_ids.unsqueeze(0),
        #         output_hidden_states=True
        #     )
        #--------------------------------非batch版本---------------------------------------

        # gen_logits = outputs.logits[0, -len(label_input_ids):, :] 
        # text_attr_pooled = gen_logits.mean(dim=0)
        # text_attr_pooled = text_attr_pooled.to(torch.float32)
        # text_attr_embed = self.lp_projector(text_attr_pooled)

        last_hidden_state = outputs.hidden_states[-1] 
        text_attr_pooled = last_hidden_state.mean(dim=1)  
        text_attr_pooled = text_attr_pooled.to(torch.float32)
        text_attr_embed = self.lp_projector(text_attr_pooled)

        if return_attr:
            return outputs.loss, text_attr_embed
        else:
            return outputs.loss


    def inference(self, sample):
        question = sample['question']
        description = self.tokenizer(sample["desc"], add_special_tokens=False)

        graph_embeds = self.encode_graphs(sample, 'test')
        graph_embeds = self.projector(graph_embeds)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))

        question_tokens = self.tokenizer(question, add_special_tokens=False)

        input_ids = description.input_ids + question_tokens.input_ids + eos_user_tokens.input_ids
        inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
        inputs_embeds = torch.cat([bos_embeds, graph_embeds, inputs_embeds], dim=0).to(self.model.device) # 后面看看这里维度拼接怎么样

        attention_mask = [1] * inputs_embeds.shape[0]
        attention_mask = torch.tensor(attention_mask).to(self.model.device)

        # print(f"inputs_embeds shape: {inputs_embeds.unsqueeze(0).shape}")
        # print(f"attention_mask shape: {attention_mask.unsqueeze(0).shape}")
        # print("inputs_embeds device:", inputs_embeds.device)
        # print("attention_mask device:", attention_mask.device)
        # print("model device:", next(self.model.parameters()).device)

        with self.maybe_autocast():
            generated_ids = self.model.generate(
                inputs_embeds=inputs_embeds.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                max_new_tokens=self.max_new_tokens,
                use_cache=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            outputs = self.model(
                inputs_embeds=inputs_embeds.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                output_hidden_states=True,
                return_dict=True
            )
        last_hidden_state = outputs.hidden_states[-1] 
        text_attr_pooled = last_hidden_state.mean(dim=1)  
        text_attr_pooled = text_attr_pooled.to(torch.float32)
        text_attr_embed = self.lp_projector(text_attr_pooled)

        text_attr = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


        edge_attr = sample['graph']['edge_attr'].to(text_attr_embed.device) 
        origin_text = sample['completion_label']
        similarity = F.cosine_similarity(text_attr_embed.expand(edge_attr.shape[0], -1), edge_attr, dim=-1)

        # print(f'***similarity: {similarity}')
        # print(f'***pred:{text_attr}')
        # print(f'***origin:{origin_text}\n')

        correct = 0
        # if any(origin_text.strip() in pred.strip() for pred in text_attr):
        if any(pred.strip() in origin_text.strip() for pred in text_attr):
            correct = 1
        # 如果没有预测出正样本，直接correct = 0
        if any(pred.strip() =='' and origin_text.strip()!= '' for pred in text_attr):
            correct = 0
        
        # print(f'***correct:{correct}\n')

        # torch.cuda.empty_cache()
        gc.collect()

        return text_attr_embed, text_attr, correct


    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
