from database.chroma_store import create_vector_store
from config import EMBEDDING_MODEL, LLM_MODEL
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from loaders.document_loader import load_documents
import json

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

llm = ChatOllama(
    model = LLM_MODEL
)

vector_store = create_vector_store(embeddings)
stored = vector_store.get()

original_documents = load_documents()



class RelevantQuestions(BaseModel):
    relevant_questions: list[str] = Field(
        description = "Three questions that can be answered using the document."
    )
class IrrelevantQuestions(BaseModel):
    irrelevant_questions: list[str] = Field(
        description = "Five questions that cannot be answered using the document."
    )
documents = stored["documents"]
def generate_relevant_questions(documents):

    relevant_questions = []
    structured_llm = llm.with_structured_output(RelevantQuestions)
    for doc in documents:
        messages = [
            SystemMessage(content=(
                "Generate three different questions that could be answered "
                "using the information in the document. "
                "The relevant questions should approach the document from different angles."
            )),
            HumanMessage(content=f"Document content: {doc}")
        ]
        response = structured_llm.invoke(messages)
        
        relevant_questions.extend(response.relevant_questions)
    return relevant_questions
def generate_irrelevant_questions(original_documents):

    irrelevant_questions = []
    structured_llm = llm.with_structured_output(IrrelevantQuestions)
    for doc in original_documents:
        messages = [
            SystemMessage(content=(
                "You are helping create training data for a RAG assistant. "
                "Generate five different questions that cannot be answered using the information in the document. "
                "These questions should be irrelevant to the document content and should not be answerable based on the information provided. "
            )),

            HumanMessage(
                content=f"Document content: {doc.page_content}"
            )
        ]
        response = structured_llm.invoke(messages)
        irrelevant_questions.extend(response.irrelevant_questions)
    return irrelevant_questions

relevant_questions = generate_relevant_questions(documents)

irrelevant_questions = generate_irrelevant_questions(original_documents)


training_data = {
    "relevant_questions": relevant_questions,
    "irrelevant_questions": irrelevant_questions
}

with open("training_data.json", "w") as file:
    json.dump(training_data, file, indent=4)