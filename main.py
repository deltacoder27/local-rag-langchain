from config import EMBEDDING_MODEL
from database.chroma_store import create_vector_store, document_scanner
from langchain_ollama import OllamaEmbeddings
from loaders.document_loader import load_documents
from langchain_core.messages import AIMessage, HumanMessage
from rag import answer_question
from guardrail import guardrail
from conversation_history import create_messages_store, loader, retrieve_relevant_memories


short_term_history = []
documents = load_documents()
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

messages_store = create_messages_store(embeddings)

vector_store = create_vector_store(embeddings)
document_scanner(documents, vector_store)


while True:
    if len(short_term_history) > 2:
        conversation = "\n".join([f"{message.type}: {message.content}" for message in short_term_history[:-2]])
        loader(conversation, messages_store)
        short_term_history = short_term_history[-2:]
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        messages_store.delete_collection()
        break
    guardrail_result = guardrail(question, short_term_history)
    if guardrail_result["Type"] == "Approved":
        answer = answer_question(guardrail_result["query"])
        print(f"Answer: {answer}")
        short_term_history.append(HumanMessage(content=question))
        short_term_history.append(AIMessage(content=answer))
        
    else:
        print("Your question was rejected by the guardrail. Please rephrase your question.")
