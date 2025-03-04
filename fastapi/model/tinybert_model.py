from model.base_embedding_model import EmbedModel
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as ort


class TinyBert(EmbedModel):
    def __init__(self, model:Path, tokenizer=str, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
    
    def load_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(kwargs.get('TOKENIZER_PATH'))

    def load_tokenizer(self, **kwargs):
        model_path = kwargs.get('MODEL_PATH')
        model_name = kwargs.get('MODEL_NAME', None)
        return ort.InferenceSession(str(model_path))
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Tokenize texts
        encoded_input = self.tokenizer(
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
        outputs = self.model.run(None, model_inputs)

        embeddings = outputs[1]
        return embeddings.tolist()

