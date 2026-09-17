import joblib

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

def ml_guardrail(question):
    question_tfidf = vectorizer.transform([question])
    prediction = model.predict(question_tfidf)
    return prediction[0]

if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        prediction = ml_guardrail(question)
        print(f"Prediction: {prediction}")