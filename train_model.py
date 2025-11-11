import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import cleaner, lemmatize_text

print("Loading dataset...")
df = pd.read_csv("database/tweets.csv", names=['sentiment', 'ids', 'date', 'flag', 'user', 'text'])

print("Preprocessing data...")
df["processed_text"] = df["text"].apply(cleaner)
df["lemmatized_text"] = df["processed_text"].apply(lemmatize_text)

x = df["lemmatized_text"]
y = df["sentiment"].astype(int)

print("Splitting data...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10, stratify=y)

print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=50000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

print("Training model...")
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver='liblinear',
    random_state=42
)
model.fit(x_train_vec, y_train)

print("Evaluating model...")
y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

print("Saving model and vectorizer...")
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model training complete! Files saved:")
print("- sentiment_model.pkl")
print("- vectorizer.pkl")