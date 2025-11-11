import pandas as pd
import re
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

def cleaner(string):
    # Remove URLs
    string = re.sub(r"http\S+|www\S+|https\S+", "", string)
    # Replace mentions with token
    string = re.sub(r"@\w+", "[USER]", string)
    # Remove hashtag symbol but keep text
    string = re.sub(r"#(\w+)", r"\1", string)
    # Remove special characters but keep emoticons
    string = re.sub(r"[^a-zA-Z0-9\s:)(:D;)\-_]", "", string)
    # Handle repeated characters (e.g., "sooooo" -> "soo")
    string = re.sub(r"(.)\1{2,}", r"\1\1", string)
    # Remove extra whitespace
    string = re.sub(r"\s+", " ", string)
    string = string.lower().strip()
    return string

def lemmatize_text(string):
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in string.split()]
    return ' '.join(words)

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