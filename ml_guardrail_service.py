import joblib
from guardrail import transform_query
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

def ml_guardrail(question):
    question_tfidf = vectorizer.transform([question])
    prediction = model.predict(question_tfidf)
    return prediction[0]
def guardrail(question, conversation_history):
    prediction = ml_guardrail(question)
    if prediction == 1:
        return "Approved"
    else:
        new_query = transform_query(question, conversation_history)
        new_prediction = ml_guardrail(new_query)
        if new_prediction == 1:
            return new_query
    return "Rejected"
if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        approval = guardrail(question, "")
        print(f"Approval: {approval}")