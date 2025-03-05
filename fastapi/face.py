from fastapi import FastAPI, HTTPException
import os
from model_utils import get_embeddings, init_model
from vectordb_utils import init_chroma_client, CLIENT_DB
from types_module import CollectionCreate, EmbeddingInput, QueryInput
from dotenv import dotenv_values

# Load environment variables as a dictionary
config = dotenv_values(".env")

app = FastAPI(title="ChromaDB API") 

# init_chroma_client()
init_model()


# API endpoints
@app.post("/collections/")
async def create_collection(collection_input: CollectionCreate):
    try:
        collection_name = CLIENT_DB.create_db(collection_input=collection_name)
        return {"message": f"Collection {collection_name} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    try:
        CLIENT_DB.delete_db(collection_name=collection_name)
        return {"message": f"Collection {collection_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/collections/{collection_name}/add")
async def add_items(collection_name: str, input_data: EmbeddingInput):
    try:
        
        embeddings = get_embeddings(input_data.texts)
        CLIENT_DB.push_item(
                    collection_name=collection_name,
                    embeddings=embeddings,
                    input_data=input_data,
                    )
        return {"message": f"Added {len(input_data.texts)} items to collection {collection_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/collections/{collection_name}/delete")
async def delete_items(collection_name: str, ids: list[str]):
    try:
        CLIENT_DB.delete_items(
            collection_name=collection_name,
            ids=ids)
        return {"message": f"Deleted items with ids {ids} from collection {collection_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collections/{collection_name}/query")
async def query_items(collection_name: str, query_input: QueryInput):
    try:
        query_embeddings = get_embeddings(query_input.texts)
        results = CLIENT_DB.query_items(
                            collection_name=collection_name,
                            query_embeddings=query_embeddings,
                            query_input=query_input
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
        return CLIENT_DB.get_db(collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/collections")
async def list_collections():
    try:
        return CLIENT_DB.get_list_db_names()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))