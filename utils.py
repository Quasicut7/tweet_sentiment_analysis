import re
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

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