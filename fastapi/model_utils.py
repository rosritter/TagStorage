from pathlib import Path
from dotenv import dotenv_values
from model.base_embedding_model import EmbedModel
from model.tinybert_model import TinyBert
from types_module import ListEntityItem
from typing import List
import numpy as np

def softmax(x: np.ndarray):
    return np.exp(x) / sum(np.exp(x))

config = dotenv_values(".env")

MODEL_PATH = Path(*Path(config.get("MODEL_PATH", '')).parts[1:])
TOKENIZER_PATH = Path(*Path(config.get("TOKENIZER_PATH", '')).parts[1:])
MODEL_NAME = config.get("MODEL_NAME", "huawei-noah/TinyBERT_General_4L_312D")
MAX_TOKENS_LENGTH = int(config.get("MAX_TOKENS_LENGTH", 512))
MODEL: EmbedModel = None

def init_model():
    # Load ONNX model and tokenizer
    global MODEL
    if MODEL_NAME in ["huawei-noah/TinyBERT_General_4L_312D",\
                      "cointegrated/rubert-tiny2"]:
        MODEL = TinyBert(
            model=MODEL_PATH, 
            tokenizer=TOKENIZER_PATH, 
            MODEL_NAME=MODEL_NAME,
            max_tokens_leght=MAX_TOKENS_LENGTH
            )
    else:
        raise NotImplementedError

def get_text_content(content: ListEntityItem):
    return content.title + "||" +\
    "||".join(content.tags) + "||" +\
    content.body

    

def get_embeddings(texts: List[ListEntityItem]) -> list[list[float]]:
    embeddings = MODEL.get_embeddings(texts=[map(get_text_content, text) for text in texts])
    return embeddings.tolist()


def get_embeddings_mean(texts: List[ListEntityItem]) -> List[List[float]]:
    # ListEntityItem example
    '''
    "
    Tags: tag_1||tag_2||..||tag_n
    Description: description_text
    etc: ...
    "
    '''
    
    output = []
    for text in texts:
        weights = []
        query = []
        if text.body:
            query.append(text.body)
            weights.append(1)
        if text.title:
            query.append(text.title)
            weights.append(1)
        if len(text.tags) > 0:
            query += text.tags
            weights.append([1 / len(text.tags)] * len(text.tags))
        
        if not len(query):
            raise Exception(f'400, Bad request no input for query')
        else:
            embeddings = MODEL.get_embeddings(query)
            normed_embeddings = [np.linalg.norm(emb, ord=2) for emb in embeddings]
            # sum(softmax([1, 1, 0.33, 0.33, 0.33])) == 1
            weights = softmax(np.array(weights)).reshape(-1, 1)
            output.append(np.sum(normed_embeddings * weights, axis=0))

    return output
