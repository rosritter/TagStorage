from pathlib import Path
from dotenv import dotenv_values
from model.base_embedding_model import EmbedModel
from model.tinybert_model import TinyBert

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

def get_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = MODEL.get_embeddings(texts=[text for text in texts])
    return embeddings.tolist()
