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

def get_text_content(content: ListEntityItem) -> str:
    parts = []
    if content.title:
        parts.append(content.title)
    if content.tags:
        parts.extend(content.tags)
    if content.body:
        parts.append(content.body)
    return " || ".join(parts)

def get_embeddings(texts: List[ListEntityItem]) -> list[list[float]]:
    # Преобразуем каждый ListEntityItem в строку
    text_contents = [get_text_content(text) for text in texts]
    embeddings = MODEL.get_embeddings(texts=text_contents)
    return embeddings.tolist()

def get_embeddings_mean(texts: List[ListEntityItem]) -> List[List[float]]:
    output = []
    for text in texts:
        query = []
        weights = []
        
        # Собираем все части текста и их веса
        if text.body:
            query.append(text.body)
            weights.append(1.0)
        if text.title:
            query.append(text.title)
            weights.append(1.0)
        if text.tags:
            query.extend(text.tags)
            # Для каждого тега добавляем уменьшенный вес
            tag_weight = 1.0 / len(text.tags) if text.tags else 0
            weights.extend([tag_weight] * len(text.tags))
        
        if not query:
            raise ValueError('No input content provided for embedding generation')
            
        # Получаем эмбеддинги для всех частей
        embeddings = MODEL.get_embeddings(query)
        # Нормализуем веса через softmax
        weights = np.array(weights)
        weights = softmax(weights)
        
        # Вычисляем взвешенное среднее эмбеддингов
        weighted_embeddings = np.array(embeddings) * weights[:, np.newaxis]
        mean_embedding = np.sum(weighted_embeddings, axis=0)
        
        # Нормализуем результирующий вектор
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
            
        output.append(mean_embedding.tolist())

    return output
