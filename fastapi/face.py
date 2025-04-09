from fastapi import FastAPI, HTTPException
from model_utils import get_embeddings, init_model, get_embeddings_mean
from vectordb_utils import init_chroma_client, CLIENT_DB
init_chroma_client()
from vectordb_utils import CLIENT_DB
from types_module import CollectionCreate, EmbeddingInput, StructuredQueryInput
from dotenv import dotenv_values
from services import add_items_t, query_items_t


# Load environment variables as a dictionary
config = dotenv_values(".env")

app = FastAPI(title="ChromaDB API") 


init_model()


# API endpoints
@app.post("/collections/")
async def create_collection(collection_input: CollectionCreate):
    try:
        print(collection_input)
        collection_name = CLIENT_DB.create_db(collection_input=collection_input)
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
async def add_items(collection_name: str, input_data: EmbeddingInput, method: str='default'):
    get_embeddings_function = get_embeddings
    if method == 'mean':
        get_embeddings_function = get_embeddings_mean
        
    return add_items_t(
            collection_name=collection_name, 
            input_data=input_data, 
            CLIENT_DB=CLIENT_DB, 
            get_embeddings=get_embeddings_function
                    )

@app.delete("/collections/{collection_name}/delete")
async def delete_items(collection_name: str, ids: EmbeddingInput):
    try:
        CLIENT_DB.delete_item(
            collection_name=collection_name,
            ids=ids)
        return {"message": f"Deleted items with ids {ids} from collection {collection_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/collections/{collection_name}/query")
async def query_items(collection_name: str, query_input: StructuredQueryInput, method: str='default'):
    get_embeddings_function = get_embeddings
    if method == 'mean':
        get_embeddings_function = get_embeddings_mean
        
    return query_items_t(
        collection_name=collection_name,
        query_input=query_input,
        CLIENT_DB=CLIENT_DB,
        get_embeddings=get_embeddings_function
    )

@app.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    try:
        return CLIENT_DB.get_db_info(collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/collections")
async def list_collections():
    try:
        collections = CLIENT_DB.get_list_db_names()
        return {
            "collections" : collections
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))