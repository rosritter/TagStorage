import time
import chromadb
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Optional
from pathlib import Path
from dotenv import dotenv_values
from urllib.parse import urlparse

# Load environment variables as a dictionary
config = dotenv_values(".env")

# Get environment variables
MODEL_PATH = Path(*Path(config.get("MODEL_PATH", '')).parts[1:])
TOKENIZER_PATH = Path(*Path(config.get("TOKENIZER_PATH", '')).parts[1:])
MODEL_NAME = config.get("MODEL_NAME", "huawei-noah/TinyBERT_General_4L_312D")
tokenizer: Optional[AutoTokenizer] = None
ort_session: Optional[ort.InferenceSession] = None

def init_model():
    # Load ONNX model and tokenizer
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if not ort_session:
        ort_session = ort.InferenceSession(str(MODEL_PATH))


def init_chroma_client(max_retries=5, retry_delay=5):
    chroma_host = config.get("CHROMA_API_ENDPOINT", "http://embeddings-chroma:8000")
    print(f"Attempting to connect to ChromaDB at: {chroma_host}")
    
    # Parse the URL properly
    parsed_url = urlparse(chroma_host)
    host = parsed_url.hostname or "embeddings-chroma"
    port = parsed_url.port or 8000
    
    print(f"Parsed host: {host}, port: {port}")
    
    for attempt in range(max_retries):
        try:
            print(f"Connection attempt {attempt + 1}/{max_retries}")
            
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chromadb.Settings(
                    chroma_client_auth_provider="chromadb.auth.disabled.DisabledAuthClientProvider",
                    anonymized_telemetry=False
                )
            )
            
            # Test basic connectivity
            print("Testing connection with heartbeat...")
            client.heartbeat()
            print("Successfully connected to ChromaDB")
            
            return client
            
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed with error: {str(e)}")
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to connect to ChromaDB after {max_retries} attempts: {str(e)}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)



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