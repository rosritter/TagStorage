from abc import ABC

class EmbedModel(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.tokenizer = self.load_tokenizer(kwargs)
        self.model = self.load_model(kwargs)

    def load_tokenizer(self, **kwargs):
        raise NotImplementedError

    def load_model(self, **kwargs):
        raise NotImplementedError
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError