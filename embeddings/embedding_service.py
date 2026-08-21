from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

def embed_text(text):
    return embeddings.embed_query(text)

def embed_chunks(chunks):
    chunk_embeddings = []
    for chunk in chunks:
        vector = embeddings.embed_query(chunk)
        chunk_embeddings.append(vector)
    return chunk_embeddings