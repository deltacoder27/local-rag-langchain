from database.chroma_store import create_vector_store
from config import EMBEDDING_MODEL, LLM_MODEL
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from database.chroma_store import get_stored_documents
from langchain_classic.retrievers import EnsembleRetriever

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)
llm = ChatOllama(
    model = LLM_MODEL
)

vector_store = create_vector_store(embeddings)

stored_documents = get_stored_documents(vector_store)


bm25_retriever = BM25Retriever.from_documents(stored_documents)
bm25_retriever.k = 3

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

hybrid_retriever = EnsembleRetriever(
    retrievers = [retriever, bm25_retriever],
    weights=[0.5,0.5]
)
'''
# Testing purposes: 

conversation_history = ["User: What is Spring Boot?",
    "Assistant: Spring Boot is a Java framework."]
'''
question = "How does Spring Boot support environment-specific configuration?"

def generate_queries(question):
    messages = [
        SystemMessage(content=(
            "Generate three different search queries that could retrieve information needed to answer the user's question."
            "Each query should approach the information from a different angle. Return only the queries, one per line."
        )),
        HumanMessage(content=f"User's question: {question}")
    ]
    response = llm.invoke(messages)
    return response.content.splitlines()

def answer_question(search_query):
    queries = generate_queries(search_query)
    all_results = []
    for query in queries:
        results = hybrid_retriever.invoke(query)
        all_results.extend(results)

    unique_results = []
    seen_chunks = set()
    for result in all_results:
        chunk_hash = result.metadata.get("chunk_content_hash")

        if chunk_hash not in seen_chunks:
            seen_chunks.add(chunk_hash)
            unique_results.append(result)





    pairs = [
        [search_query, doc.page_content]
        for doc in unique_results
    ]

    scores = reranker.predict(pairs)
    scored_results = list(zip(unique_results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    reranked_results = scored_results[:3]
    #filtered_results = [result for result in reranked_results if result[1] >= 3.5]

    context = "\n".join(result.page_content for result, score in reranked_results)

    messages = [
        SystemMessage(
            content=(
                "Use the provided context to answer the user's question. "
                "If the answer cannot be found in the context, say you don't know."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {search_query}\n\n"
                f"Context:\n{context}"
            )
        )
    ]

    response = llm.invoke(messages)
    return response.content


