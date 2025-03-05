from vectorDB.base_db import VectorDB
from vectorDB.chroma_db import ChromaDB 
from dotenv import dotenv_values

# Load environment variables as a dictionary
config = dotenv_values(".env")

VECTORDB_NAME = config.get("VECTORDB_NAME", "CHROMADB")
CLIENT_DB: VectorDB = None

def init_chroma_client():
    global CLIENT_DB, config
    if VECTORDB_NAME.upper() == "CHROMADB":
        CLIENT_DB = ChromaDB(**config)
    else:
        raise NotImplementedError
