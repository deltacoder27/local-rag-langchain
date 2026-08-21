from langchain_chroma import Chroma
from langchain_core.documents import Document
import hashlib
from loaders.document_loader import load_documents, clean_text
from embeddings.embedding_service import embed_chunks
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utils.math import cosine_similarity
from chunking.semantic_chunker import create_semantic_chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    strip_whitespace=True
)

def create_vector_store(embeddings):
    vector_store = Chroma(
    collection_name = "documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
    return vector_store
    

def create_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_stored_documents(vector_store):
    stored = vector_store.get()
    return [
        Document(
            page_content=page_content,
            metadata=metadata
        )
        for page_content, metadata in zip(stored["documents"], stored["metadatas"])
    ]

def document_scanner(current_documents, vector_store):
    current_doc_title_hashes = {create_hash(Path(doc.metadata["source"]).stem): doc for doc in current_documents}
    stored_documents = get_stored_documents(vector_store)

    stored_doc_title_hashes = {
        doc.metadata.get("doc_title_hash"): doc
        for doc in stored_documents
    }


    # Check for new or updated documents
    for hash_value, current_doc in current_doc_title_hashes.items():
        if hash_value not in stored_doc_title_hashes:
            scan_value = "Load Document"  # New document found
            sync_documents(current_doc, vector_store, scan_value)
        elif hash_value in stored_doc_title_hashes and create_hash(clean_text(current_doc)) != stored_doc_title_hashes[hash_value].metadata.get("doc_content_hash"):
            scan_value = "Update Document"  # Updated document found
            sync_documents(current_doc, vector_store, scan_value)

    # Check for deleted documents
    for hash_value in stored_doc_title_hashes:
        if hash_value not in current_doc_title_hashes:
            scan_value = "Delete Document"  # Deleted document found
            sync_documents(hash_value, vector_store, scan_value)

    return "Synced"  # All documents are synced

def doc_loader(document, name):
    cleaned_text = clean_text(document)
    chunks = splitter.split_text(cleaned_text)
    chunk_embeddings = embed_chunks(chunks)
    chunk_list = create_semantic_chunks(chunk_embeddings, chunks)
    documents_to_add = []
    for i, chunk in enumerate(chunk_list):
        documents_to_add.append(Document(
            page_content=chunk,
            metadata={
                "name": name + "_" + str(i),
                "doc_title_hash": create_hash(name),
                "doc_content_hash": create_hash(cleaned_text),
                "chunk_content_hash": create_hash(chunk)
            }
        ))
    return documents_to_add


def sync_documents(document, vector_store, scan_value):
    
    if scan_value == "Load Document":
        name = Path(document.metadata["source"]).stem
        documents_to_add = doc_loader(document, name)
        print("Loader was called for document:", name)
        vector_store.add_documents(documents_to_add)
       
    if scan_value == "Update Document":
        name = Path(document.metadata["source"]).stem
        print("Update was called for document:", name)
        vector_store.delete(
            where={"doc_title_hash": create_hash(name)}
        )
        documents_to_add = doc_loader(document, name)
        vector_store.add_documents(documents_to_add)
    if scan_value == "Delete Document":
        name = vector_store.get(where={"doc_title_hash": document})["metadatas"][0]["name"]
        print("Delete was called for document:", name)
        vector_store.delete(
            where={"doc_title_hash": document}
        )



