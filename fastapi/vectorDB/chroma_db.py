from vectorDB.base_db import VectorDB
import time
import chromadb
from urllib.parse import urlparse
import numpy as np

class ChromaDB(VectorDB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_client(self, **kwargs):
        max_retries=kwargs.get(max_retries, 5) 
        retry_delay=kwargs.get(retry_delay, 5)

        chroma_host = kwargs.get("CHROMA_API_ENDPOINT", "http://chroma:8000")
        print(f"Attempting to connect to ChromaDB at: {chroma_host}")
        
        # Parse the URL properly
        parsed_url = urlparse(chroma_host)
        host = parsed_url.hostname or "chroma"
        port = parsed_url.port or 8000
        
        print(f"Parsed host: {host}, port: {port}")
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        for attempt in range(max_retries):
            try:
                print(f"Connection attempt {attempt + 1}/{max_retries}")
                
                client = chromadb.HttpClient(host, port)
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

    def delete_db(self, **kwargs):
        raise NotImplementedError
    
    def create_db(self, **kwargs):
        raise NotImplementedError
    
    def get_list_db_names(self, **kwargs):
        return self.client.list_collections()
    
    def get_db(self, collection_name:str, **kwargs):
        collection = self.client.get_collection(collection_name)
        return {
            "name": collection.name,
            "metadata": collection.metadata,
            "count": collection.count()
        }
    def push_item(self, **kwargs):
        raise NotImplementedError
        
    def delete_item(self, **kwargs):
        raise NotImplementedError

    def query_items(self, collection_name:str, 
                    query_embeddings:np.ndarray,
                    query_input, 
                    **kwargs):
        collection = self.get_db(collection_name=collection_name,
                                 kwargs=kwargs)
        
        return collection.query(
                                query_embeddings=query_embeddings,
                                n_results=query_input.n_results,
                                include=['documents', 'metadatas', 'distances']
                                )