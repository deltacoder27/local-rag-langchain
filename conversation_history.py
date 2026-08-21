from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_messages_store(embeddings):
    messages_store = Chroma(
    collection_name = "messages",
    embedding_function=embeddings,
    persist_directory="./chroma_db_messages"
)
    return messages_store

def loader(conversation, messages_store):
    stored = messages_store.get()
    messages_store.add_documents([
        Document(
            page_content=conversation,
            id=str(len(stored["documents"]) + 1)
        )
    ])

def retrieve_relevant_memories(query, messages_store):
    result = messages_store.similarity_search_with_score(query, k=2)
    threshold = 1.0
    allowed_documents = []
    for doc, score in result:
        if score <= threshold:
            allowed_documents.append(doc.page_content)
    return allowed_documents

