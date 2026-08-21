from conversation_history import create_messages_store
from database.chroma_store import create_vector_store
from config import EMBEDDING_MODEL, LLM_MODEL
from langchain_ollama import OllamaEmbeddings, ChatOllama
from sentence_transformers import CrossEncoder
from langchain_core.messages import SystemMessage, HumanMessage

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

messages_store = create_messages_store(embeddings)

vector_store = create_vector_store(embeddings)
'''
# Testing purposes
conversation_history = [
    "User: What is Spring Boot?",
    "Assistant: Spring Boot is a Java framework."
]
query = "What is that"
'''
def transform_query(query, conversation_history):
    messages = [
        SystemMessage(content=(
                "Rewrite the user's latest question into a standalone "
                "question that can be understood without the conversation history. "
                "Preserve the original meaning. "
                "If the question is already standalone, return it unchanged. "
                "Return only the rewritten question."
            )),
        HumanMessage(content=f"Conversation history: {conversation_history}\n\nUser's latest question: {query}")
    ]
    llm = ChatOllama(model=LLM_MODEL)
    response = llm.invoke(messages)
    return response.content
def transformed_query_check(query, store):
    print("Transformed query: ", query)
    result = store.similarity_search_with_score(query, k=3)
    threshold = 1.0
    allowed_documents = []
    for doc, score in result:
        print("Transformed similarity score: ", score)
        if score <= threshold:
            allowed_documents.append(doc.page_content)
    return allowed_documents

def guardrail(query, conversation_history):
    guardrail_check = {}
    result = vector_store.similarity_search_with_score(query, k=3)
    threshold = 1.0
    allowed_documents = []
    for doc, score in result:
        print("Original similarity score: ", score)
        if score <= threshold:
            allowed_documents.append(doc.page_content)

    if allowed_documents == []:
        query = transform_query(query, conversation_history)
        allowed_documents = transformed_query_check(query, vector_store)
    if allowed_documents == []:
        query = transform_query(query, messages_store.get())
        allowed_documents = transformed_query_check(query, messages_store)
    if allowed_documents != []:
        pairs = [
            [query, doc]
            for doc in allowed_documents
        ]

        scores = reranker.predict(pairs)
        scored_results = list(zip(allowed_documents, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        reranked_results = scored_results[:3]
        approved_results = []
        for doc, score in reranked_results:
            print("Reranked score: ", score)
            if score > 0:
                approved_results.append(doc)
                guardrail_check["Type"] = "Approved"
                guardrail_check["documents"] = approved_results
                guardrail_check["query"] = query
                return guardrail_check
    guardrail_check["Type"] = "Rejected"
    return guardrail_check

