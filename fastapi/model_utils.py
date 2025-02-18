import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List
from pathlib import Path
from dotenv import dotenv_values
import time
import chromadb

# Load environment variables as a dictionary
config = dotenv_values(".env")

# Get environment variables
MODEL_PATH = Path(*Path(config.get("MODEL_PATH", '')).parts[1:])
TOKENIZER_PATH = Path(*Path(config.get("TOKENIZER_PATH", '')).parts[1:])
MODEL_NAME = config.get("MODEL_NAME", "huawei-noah/TinyBERT_General_4L_312D")




# Initialize ChromaDB client with retry logic
def init_chroma_client(max_retries=5, retry_delay=5):
    CHROMA_API_ENDPOINT = config.get("CHROMA_API_ENDPOINT", "http://embeddings-chroma:8000")
    
    for attempt in range(max_retries):
        try:
            client = chromadb.HttpClient(host=CHROMA_API_ENDPOINT)
            # Try to get or create default tenant
            try:
                client.get_tenant("default_tenant")
            except Exception:
                try:
                    client.create_tenant("default_tenant")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        raise
            return client
        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to connect to ChromaDB after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)




# Load ONNX model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
ort_session = ort.InferenceSession(MODEL_PATH)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    # Tokenize texts
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='np'
    )
    
    # Get embeddings from ONNX model
    model_inputs = {
        'input_ids': encoded_input['input_ids'],
        'attention_mask': encoded_input['attention_mask']
    }
    outputs = ort_session.run(None, model_inputs)
    
    # Get embeddings from the second output (index 1)
    embeddings = outputs[1]
    return embeddings.tolist()