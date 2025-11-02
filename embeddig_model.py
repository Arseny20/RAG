from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import numpy as np
from langchain.embeddings.base import Embeddings
import torch.nn.functional as F
from tqdm import tqdm
from config import CONFIG
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Embeddings(Embeddings):
    def __init__(self, model_name:str, batch_size: int = 8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(f"./models/tokenizer/{model_name}")
        self.model = AutoModel.from_pretrained(f"./models/{model_name}").to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.dim = self.model.config.hidden_size
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        input_text = "query: " + text
        input_dict = self.tokenizer(input_text, 
                                        return_tensors=CONFIG['tokenizer']['return_tensors'],
                                        padding=CONFIG['tokenizer']['padding'],
                                        max_length = CONFIG['tokenizer']['max_length'],
                                        truncation=CONFIG['tokenizer']['truncation']).to(self.device)
        model_output = self.model(**input_dict)
        embeddings = average_pool(model_output.last_hidden_state, input_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            input_texts = texts[i:i + self.batch_size]
            batch_dict = self.tokenizer(input_texts,
                                        return_tensors=CONFIG['tokenizer']['return_tensors'],
                                        padding=CONFIG['tokenizer']['padding'],
                                        max_length = CONFIG['tokenizer']['max_length'],
                                        truncation=CONFIG['tokenizer']['truncation']).to(self.device)
            model_output = self.model(**batch_dict)

            embeddings = average_pool(model_output.last_hidden_state, batch_dict["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings
