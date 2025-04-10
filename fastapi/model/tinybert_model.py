from model.base_embedding_model import EmbedModel
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as ort


class TinyBert(EmbedModel):
    def __init__(self, model:Path, tokenizer=str, max_tokens_leght:int=512, **kwargs):
        max_tokens_leght = max_tokens_leght if 0 < max_tokens_leght < 513 else 512
        super().__init__(model=model, 
                         tokenizer=tokenizer, 
                         max_tokens_leght=max_tokens_leght, 
                         **kwargs)

    def load_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(kwargs.get('tokenizer'))

    def load_model(self, **kwargs):
        model_path = kwargs.get('model')
        model_name = kwargs.get('MODEL_NAME', None)
        return ort.InferenceSession(str(model_path))

    def get_max_tokens_lenght(self, **kwargs):
        return kwargs.get('max_tokens_leght', 512)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,    # :TODO depend max_length from model2model
            return_tensors='np'
        )
        # Get embeddings from ONNX model
        model_inputs = {
            'input_ids': encoded_input['input_ids'],
            'attention_mask': encoded_input['attention_mask']
        }
        outputs = self.model.run(None, model_inputs)

        embeddings = outputs[1]
        return embeddings
