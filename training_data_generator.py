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


original_documents = load_documents()



class RelevantQuestions(BaseModel):
    relevant_questions: list[str] = Field(
        description = "Five questions that can be answered using the document."
    )
class IrrelevantQuestions(BaseModel):
    irrelevant_questions: list[str] = Field(
        description = "Five questions that cannot be answered using the document."
    )

def generate_relevant_questions(documents):

    relevant_questions = []
    structured_llm = llm.with_structured_output(RelevantQuestions)
    for doc in documents:
        messages = [
            SystemMessage(content=(
                "Generate five different questions that can be answered using the information in the document. "
                "Make the questions meaningfully different from each other and vary how the information is asked about. "
                "Include a mix of: "
                "1. Direct factual questions "
                "2. Questions about people, entities, events, or subjects when applicable "
                "3. Questions about relationships, characteristics, or responsibilities when applicable" 
                "4. Questions about processes, explanations, or how something works when applicable "
                "5. Questions using alternative wording or synonyms "
                "6. Natural questions a user might realistically ask "
                "7. Short questions when appropriate "
                "Use these categories as guidance; the three questions do not need to cover every category. "
                "Only ask questions that can actually be answered using information contained in the document. Do not require outside knowledge to answer the questions. "
            )),
            HumanMessage(content=f"Document content: {doc.page_content}")
        ]
        response = structured_llm.invoke(messages)
        print("Generated:", response.relevant_questions)
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
                "Make the questions meaningfully different from each other and vary how the information is asked about. "
                "Make sure all five questions are distinct and do not repeat the same question or ask essentially the same question using slightly different wording. "
                "Include a mix of: "
                "1. Clearly unrelated questions "
                "2. General knowledge questions that are not answered by the document "
                "3. Questions that use words, concepts, or terminology from the document but ask about information not contained in it "
                "4. Questions that may appear related to the document but cannot actually be answered using its information "
                "5. Natural questions a user might realistically ask but that the document does not provide an answer to "
                "The goal is to create challenging irrelevant examples rather than only obviously unrelated questions. "
                "Make sure every question is genuinely unanswerable using the information provided in the document."
            )),

            HumanMessage(
                content=f"Document content: {doc.page_content}"
            )
        ]
        response = structured_llm.invoke(messages)
        irrelevant_questions.extend(response.irrelevant_questions)
    return irrelevant_questions

relevant_questions = generate_relevant_questions(original_documents)

irrelevant_questions = generate_irrelevant_questions(original_documents)


training_data = {
    "relevant_questions": relevant_questions,
    "irrelevant_questions": irrelevant_questions
}
print("Relevant questions:", len(relevant_questions))
print("Irrelevant questions:", len(irrelevant_questions))

with open("training_data.json", "w") as file:
    json.dump(training_data, file, indent=4)