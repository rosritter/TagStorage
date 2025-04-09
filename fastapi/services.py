from types_module import CollectionCreate, EmbeddingInput, StructuredQueryInput
from fastapi import FastAPI, HTTPException
EPS = 1e-10

def query_items_t(collection_name: str, query_input: StructuredQueryInput, CLIENT_DB , get_embeddings: callable):
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
                [1 - (d / (max_distance + EPS)) for d in distances]
                for distances in results['distances']
            ]
            
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

def add_items_t(collection_name: str, input_data: EmbeddingInput,  CLIENT_DB , get_embeddings: callable):
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