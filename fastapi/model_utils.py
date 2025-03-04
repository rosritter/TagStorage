from pathlib import Path
from dotenv import dotenv_values
from model.base_embedding_model import EmbedModel
from model.tinybert_model import TinyBert

config = dotenv_values(".env")

MODEL_PATH = Path(*Path(config.get("MODEL_PATH", '')).parts[1:])
TOKENIZER_PATH = Path(*Path(config.get("TOKENIZER_PATH", '')).parts[1:])
MODEL_NAME = config.get("MODEL_NAME", "huawei-noah/TinyBERT_General_4L_312D")
MODEL: EmbedModel

def init_model():
    # Load ONNX model and tokenizer
    global MODEL
    if MODEL_NAME == "huawei-noah/TinyBERT_General_4L_312D":
        MODEL = TinyBert(
            model=MODEL_PATH, 
            tokenizer=TOKENIZER_PATH, 
            MODEL_NAME=MODEL_NAME
            )
    else:
        raise NotImplementedError

def get_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = MODEL.get_embeddings(texts=texts)
    return embeddings.tolist()