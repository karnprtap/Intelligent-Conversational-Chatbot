import os
import random
import streamlit as st
import pandas as pd
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ---------------------------
# Download NLTK Data
# ---------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------------------------
# Load Dataset and Model
# ---------------------------
df = pd.read_csv("chatbot_human_interaction.csv")
df = df.dropna(subset=["text", "intent", "response"])

model = pickle.load(open("chatbot_model.pkl", "rb"))

# Change to vectorizer.pkl if you rename the file
vectorizer = pickle.load(open("vectorizer.pk1", "rb"))

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

# ---------------------------
# Chatbot Response Function
# ---------------------------
def get_response(user_input):
    clean = clean_text(user_input)

    vector = vectorizer.transform([clean])

    intent = model.predict(vector)[0]

    matching = df[df["intent"] == intent]["response"].tolist()
    response = random.choice(matching) if matching else "Sorry, I didn't understand that."

    return response

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📋 Menu Navigation")

menu = st.sidebar.radio(
    "Select Option",
    ["Home", "About", "Chatbot"]
)
st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #0B3D91;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white;
}

/* Radio button labels */
[data-testid="stSidebar"] .stRadio label {
    color: white !important;
}

/* Sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Home Page
# ---------------------------
if menu == "Home":

    st.title("🤖 Intelligent Conversational Chatbot")

    st.write("""
Welcome to the **Intelligent Conversational Chatbot**.

This chatbot uses **Machine Learning** and **Natural Language Processing (NLP)** to understand user queries and provide relevant responses.

Use the sidebar to navigate through the application.
""")

# ---------------------------
# About Page
# ---------------------------
elif menu == "About":

    st.title("ℹ️ About")

    st.markdown("""
## Intelligent Conversational Chatbot

The Intelligent Conversational Chatbot is a Machine Learning and Natural Language Processing (NLP) based web application that understands user queries and provides accurate, intent-based responses in real time.

The chatbot uses **TF-IDF Vectorization** and a trained **Machine Learning model** to classify user intents and generate meaningful responses.

### 🚀 Key Features

- 🤖 Intelligent intent recognition
- 💬 Real-time conversation
- 📝 NLP-based preprocessing
- 🔍 TF-IDF feature extraction
- ⚡ Fast response generation
- 🎨 User-friendly Streamlit interface
- 🔄 Easy to retrain with new datasets

### 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Pickle

### 📌 Project Description

This project demonstrates the practical implementation of Machine Learning and NLP to automate conversations for customer support, education, FAQs, and other conversational applications.
""")

# ---------------------------
# Chatbot Page
# ---------------------------
elif menu == "Chatbot":

    st.title("🤖 Chatbot")

    user_input = st.text_input("Ask me something:")

    if st.button("Send"):

        if user_input.strip() == "":
            st.warning("Please enter a message.")

        else:
            response = get_response(user_input)

            st.success("Bot:")
            st.write(response)