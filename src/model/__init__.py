from src.model.llm import LLM
from src.model.graph_llm import GraphLLM
from src.model.graph_llm_v3 import CompletionModel
from src.model.LP import LinkPredictionModel
from src.model.llm_rag import GraphLLM as LLM_rag



load_model = {
    'llm': LLM,
    'inference_llm': LLM_rag,
    'graph_llm': GraphLLM,
    'LP':LinkPredictionModel,
    'graph_llm_v3': CompletionModel
}

# Replace the following with the model paths
llama_model_path = {
    '7b': 'meta-llama/Llama-2-7b-hf',
    '13b': 'meta-llama/Llama-2-13b-hf',
    '8b' : 'meta-llama/Llama-3.1-8B',
    '7b_chat': 'meta-llama/Llama-2-7b-chat-hf',
    '13b_chat': 'meta-llama/Llama-2-13b-chat-hf',
}
