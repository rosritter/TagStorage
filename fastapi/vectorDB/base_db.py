from abc import ABC


class VectorDB(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.client = self.load_client(**kwargs)

    def load_client(self, **kwargs):
        raise NotImplementedError
    
    def delete_db(self, **kwargs):
        raise NotImplementedError
    
    def create_db(self, **kwargs):
        raise NotImplementedError
    
    def get_list_db_names(self, **kwargs):
        raise NotImplementedError
    
    def get_db(self, **kwargs):
        raise NotImplementedError
    
    def push_item(self, **kwargs):
        raise NotImplementedError
        
    def delete_item(self, **kwargs):
        raise NotImplementedError

    def query_items(self, **kwargs):
        raise NotImplementedError