from vectorDB.base_db import VectorDB
import time
import chromadb
from urllib.parse import urlparse
import numpy as np
from types_module import CollectionCreate, EmbeddingInput
from model_utils import get_text_content

class ChromaDB(VectorDB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_client(self, **kwargs):
        max_retries=kwargs.get("max_retries", 5) 
        retry_delay=kwargs.get("retry_delay", 5)

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

    def delete_db(self, collection_name:str, **kwargs):
        self.client.delete_collection(collection_name)
    
    def create_db(self, collection_input:CollectionCreate, **kwargs) -> str:
        collection = self.client.create_collection(
                            name=collection_input.name,
                            # metadata=collection_input.metadata
                        )
        return collection.name
    
    def get_list_db_names(self, **kwargs):
        collections = self.client.list_collections()
        return collections
    
    def get_db(self, collection_name:str, **kwargs):
        collection = self.client.get_collection(collection_name)
        return collection

    def get_db_info(self, collection_name:str, **kwargs):
        collection = self.get_db(collection_name)
        return {
            "name": collection.name,
            # "metadata": collection.metadata,
            "count": collection.count()
        }

    def push_item(self, 
                  collection_name:str,
                  embeddings:np.ndarray,
                  input_data: EmbeddingInput,
                  **kwargs):
        collection = self.get_db(collection_name)
        # Преобразуем ListEntityItem в строки
        documents = [get_text_content(text) for text in input_data.texts]
        collection.add(
            embeddings=embeddings,
            documents=documents,  # Используем преобразованные строки
            ids=input_data.ids,
            # metadatas=input_data.metadatas
        )
        
    def delete_item(self, collection_name, ids, **kwargs) -> None:
        collection = self.get_db(collection_name)
        collection.delete(ids=ids)

    def query_items(self, collection_name:str, 
                    query_embeddings:np.ndarray,
                    query_input, 
                    **kwargs):
        collection = self.get_db(collection_name=collection_name,
                                 **kwargs)
        # Преобразуем ListEntityItem в строки, если они есть в запросе
        documents = [get_text_content(text) for text in query_input.texts] if hasattr(query_input.texts[0], 'body') else query_input.texts
        
        return collection.query(
                                query_embeddings=query_embeddings,
                                n_results=query_input.n_results,
                                include=[
                                    'documents', 
                                    # 'metadatas', 
                                    'distances'
                                    ]
                                )