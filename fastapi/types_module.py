from typing import List, Dict, Optional
from pydantic import BaseModel

# Pydantic models for request validation
class CollectionCreate(BaseModel):
    name: str
    metadata: Optional[Dict] = None

class EmbeddingInput(BaseModel):
    texts: List[str]
    ids: List[str]
    metadatas: Optional[List[Dict]] = None

class QueryInput(BaseModel):
    texts: List[str]
    n_results: int = 5
