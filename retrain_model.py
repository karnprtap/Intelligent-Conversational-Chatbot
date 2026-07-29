"""
Retrain the chatbot model with consistent preprocessing.
Run this script to regenerate chatbot_model.pkl and vectorizer.pk1.
"""
import pandas as pd
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Load and clean data
df = pd.read_csv("chatbot_human_interaction.csv")
df = df.dropna(subset=["text", "intent", "response"])
print(f"Loaded {len(df)} rows after dropping NaN")

# Setup preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["clean_text"])
y = df["intent"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save model and vectorizer
pickle.dump(model, open("chatbot_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pk1", "wb"))
print("Model and vectorizer saved successfully!")
