from config import EMBEDDING_MODEL
from database.chroma_store import create_vector_store, document_scanner
from langchain_ollama import OllamaEmbeddings
from loaders.document_loader import load_documents
from langchain_core.messages import AIMessage, HumanMessage
from rag import answer_question
from guardrail import guardrail


conversation_history = []
documents = load_documents()
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = create_vector_store(embeddings)
document_scanner(documents, vector_store)


while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    guardrail_result = guardrail(question, conversation_history)
    if guardrail_result["Type"] == "Approved":
        answer = answer_question(guardrail_result["query"])
        print(f"Answer: {answer}")
        conversation_history.append(HumanMessage(content=question))
        conversation_history.append(AIMessage(content=answer))
    else:
        print("Your question was rejected by the guardrail. Please rephrase your question.")
