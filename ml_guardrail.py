import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
with open('training_data.json', 'r') as file:
    training_data = json.load(file)

    relevant_questions = training_data['relevant_questions']
    irrelevant_questions = training_data['irrelevant_questions']

X = relevant_questions + irrelevant_questions
y = [1] * len(relevant_questions) + [0] * len(irrelevant_questions)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")  

print(classification_report(
    y_test,
    predictions,
    target_names=["Irrelevant", "Relevant"]
))

confusion = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:\n", confusion)

joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')