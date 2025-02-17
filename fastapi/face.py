from fastapi import FastAPI, HTTPException
from typing import List
import chromadb
from chromadb.utils import embedding_functions
import os
from model_utils import get_embeddings
from types import CollectionCreate, EmbeddingInput, QueryInput
from dotenv import dotenv_values

# Load the .env file into a dictionary
config = dotenv_values(".env")

app = FastAPI(title="ChromaDB API")

# Initialize ChromaDB client
CHROMA_API_ENDPOINT = os.getenv("CHROMA_API_ENDPOINT", "http://chroma:8000")
client = chromadb.HttpClient(host=CHROMA_API_ENDPOINT)



# API endpoints
@app.post("/collections/")
async def create_collection(collection_input: CollectionCreate):
    try:
        collection = client.create_collection(
            name=collection_input.name,
            metadata=collection_input.metadata
        )
        return {"message": f"Collection {collection_input.name} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    try:
        client.delete_collection(collection_name)
        return {"message": f"Collection {collection_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/collections/{collection_name}/add")
async def add_items(collection_name: str, input_data: EmbeddingInput):
    try:
        collection = client.get_collection(collection_name)
        embeddings = get_embeddings(input_data.texts)
        
        collection.add(
            embeddings=embeddings,
            documents=input_data.texts,
            ids=input_data.ids,
            metadatas=input_data.metadatas
        )
        return {"message": f"Added {len(input_data.texts)} items to collection {collection_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/collections/{collection_name}/delete")
async def delete_items(collection_name: str, ids: List[str]):
    try:
        collection = client.get_collection(collection_name)
        collection.delete(ids=ids)
        return {"message": f"Deleted items with ids {ids} from collection {collection_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collections/{collection_name}/query")
async def query_items(collection_name: str, query_input: QueryInput):
    try:
        collection = client.get_collection(collection_name)
        query_embeddings = get_embeddings(query_input.texts)
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=query_input.n_results,
            include=['documents', 'metadatas', 'distances']  # Include distances for scores
        )
        
        # Convert distances to similarity scores (1 - normalized_distance)
        if 'distances' in results:
            max_distance = max(max(distances) for distances in results['distances'])
            results['scores'] = [
                [1 - (d / max_distance) for d in distances]
                for distances in results['distances']
            ]
            
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    try:
        collection = client.get_collection(collection_name)
        return {
            "name": collection.name,
            "metadata": collection.metadata,
            "count": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/collections")
async def list_collections():
    try:
        collections = client.list_collections()
        return {"collections": [
            {"name": col.name, "metadata": col.metadata} 
            for col in collections
        ]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))