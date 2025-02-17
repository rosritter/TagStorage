import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List
from dotenv import load_dotenv

# Load environment variables
config = load_dotenv()

# Get environment variables
MODEL_PATH = config.get("MODEL_PATH", '')
TOKENIZER_PATH = config.get("TOKENIZER_PATH", '')
MODEL_NAME = config.get("MODEL_NAME", "huawei-noah/TinyBERT_General_4L_312D")

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