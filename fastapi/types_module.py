from typing import List, Dict, Optional
from pydantic import BaseModel

# Pydantic models for request validation
class CollectionCreate(BaseModel):
    name: str
    metadata: Optional[Dict] = None

class ListEntityItem(BaseModel):
    body: str = ""
    title: str = ""
    tags: List[str] = []

class StructuredQueryInput(BaseModel):
    texts: List[ListEntityItem]
    n_results: int = 5 

class EmbeddingInput(BaseModel):
    texts: List[ListEntityItem]
    ids: List[str]
    metadatas: Optional[List[Dict]] = None
