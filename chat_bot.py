import os
import json
import ssl
import random
import streamlit as st
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK download fix ---
ssl._create_default_https_context = ssl._create_unverified_context

def safe_nltk_download(resource, path):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)
        try:
            nltk.data.find(path)
        except LookupError:
            st.error(f"Failed to download NLTK resource: {resource}")

safe_nltk_download("punkt", "tokenizers/punkt")
safe_nltk_download("stopwords", "corpora/stopwords")
safe_nltk_download("wordnet", "corpora/wordnet")

# --- Preprocessing ---
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(tokens)

# --- Load intents ---
file_path = "tech_intents.json"
intents = []
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if "intents" in data:
            intents = data["intents"]
except Exception as e:
    st.error(f"Error loading intents: {e}")

# --- Preprocess training data ---
patterns, responses, tags, processed_patterns = [], [], [], []
if intents:
    for intent in intents:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            processed_patterns.append(preprocess(pattern))
            tags.append(intent["tag"])
            responses.append(random.choice(intent["responses"]))

# --- Train TF-IDF model ---
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(processed_patterns) if processed_patterns else None

# --- Log chat ---
def log_chat(user_input, bot_response):
    log_file = "tech_chat_log.csv"
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User Input": user_input,
        "Bot Response": bot_response
    }
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(log_file, index=False)

# --- Chatbot logic ---
def chatbot(input_text):
    if x_train is None:
        return "Sorry, I can't process your query right now."
    input_vec = vectorizer.transform([preprocess(input_text)])
    similarity = cosine_similarity(input_vec, x_train).flatten()
    best_index = np.argmax(similarity)
    confidence = similarity[best_index]

    if confidence < 0.3:
        return "I'm not sure I understand that. Can you rephrase?"

    matched_tag = tags[best_index]
    matched_intent = next((intent for intent in intents if intent["tag"] == matched_tag), None)
    if matched_intent:
        return random.choice(matched_intent["responses"])
    return "I understand your question, but I don't have a good answer yet."

# --- Streamlit UI ---
def main():
    st.title("💻 Tech Support Chatbot")
    st.sidebar.title("Menu")
    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Go to", menu)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if choice == "Chat":
        st.subheader("Ask Your Tech Questions")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["text"])

        user_input = st.chat_input("Type your question here...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            bot_reply = chatbot(user_input)
            st.session_state.chat_history.append({"role": "assistant", "text": bot_reply})
            with st.chat_message("assistant"):
                st.write(bot_reply)

            log_chat(user_input, bot_reply)

    elif choice == "Conversation History":
        st.subheader("Past Conversations")
        search_term = st.text_input("Search conversation:")
        for msg in st.session_state.chat_history:
            if search_term.lower() in msg["text"].lower():
                st.text(f"{msg['role'].title()}: {msg['text']}")

    elif choice == "About":
        st.subheader("About This Chatbot")
        st.write("""
        🤖 This is a smart tech support chatbot built using:
        - Streamlit for the UI
        - NLTK for text processing (lemmatization, tokenization)
        - Scikit-learn for TF-IDF + cosine similarity matching

        ✨ It helps answer basic technical questions and can be easily extended with more intents.
        """)

if __name__ == "__main__":
    main()
