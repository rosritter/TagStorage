from types_module import CollectionCreate, EmbeddingInput, StructuredQueryInput, ListEntityItem
from fastapi import FastAPI, HTTPException
from model_utils import get_text_content
from vectorDB.base_db import VectorDB
EPS = 1e-10

def query_items_t(collection_name: str, query_input: StructuredQueryInput, CLIENT_DB: VectorDB, get_embeddings: callable):
    try:
        # Получаем эмбеддинги для структурированных данных
        query_embeddings = get_embeddings(query_input.texts)
        
        # Преобразуем ListEntityItem в строки для запроса и создаем новые ListEntityItem
        query_documents = []
        for text in query_input.texts:
            content = get_text_content(text)
            query_documents.append(ListEntityItem(body=content))
        
        # Создаем стандартный запрос для ChromaDB
        results = CLIENT_DB.query_items(
            collection_name=collection_name,
            query_embeddings=query_embeddings,
            query_input=StructuredQueryInput(
                texts=query_documents,  # Используем список ListEntityItem
                n_results=query_input.n_results
            )
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
    

def add_items_t(collection_name: str, input_data: EmbeddingInput, CLIENT_DB: VectorDB, get_embeddings: callable):
    try:
        # Получаем эмбеддинги для структурированных данных
        embeddings = get_embeddings(input_data.texts)
        
        # Преобразуем ListEntityItem в строки для ChromaDB и создаем новые ListEntityItem
        documents = []
        for text in input_data.texts:
            content = get_text_content(text)
            # Создаем новый ListEntityItem только с текстовым содержимым
            documents.append(ListEntityItem(body=content))
        
        # Создаем новый input_data с преобразованными документами
        modified_input = EmbeddingInput(
            texts=documents,  # Теперь передаем список ListEntityItem
            ids=input_data.ids,
            metadatas=input_data.metadatas
        )
        
        # Сохраняем в ChromaDB
        CLIENT_DB.push_item(
            collection_name=collection_name,
            embeddings=embeddings,
            input_data=modified_input
        )
        
        return {"message": f"Added {len(input_data.texts)} items to collection {collection_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))