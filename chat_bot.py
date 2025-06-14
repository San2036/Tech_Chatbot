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

# NLTK download fix
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents
file_path = "tech_intents.json"
intents = []
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if "intents" in data:
            intents = data["intents"]
except Exception as e:
    st.error(f"Error loading intents: {e}")

# Preprocess training data
patterns, responses, tags = [], [], []
if intents:
    for intent in intents:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            responses.append(random.choice(intent["responses"]))
            tags.append(intent["tag"])

# Train TF-IDF model
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(patterns) if patterns else None

# Log chat
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

# Chatbot logic
def chatbot(input_text):
    if x_train is None:
        return "Sorry, I can't process your query right now."
    input_vec = vectorizer.transform([input_text])
    similarity = cosine_similarity(input_vec, x_train).flatten()
    best_index = np.argmax(similarity)
    confidence = similarity[best_index]
    if confidence < 0.3:
        return "I'm not sure I understand that. Can you rephrase?"
    return responses[best_index]

# Streamlit UI
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
        for msg in st.session_state.chat_history:
            st.text(f"{msg['role'].title()}: {msg['text']}")

    elif choice == "About":
        st.subheader("About This Chatbot")
        st.write("""
        This is a simple tech support chatbot built using Streamlit, NLTK, and Scikit-learn.
        It helps answer common technical issues and software-related questions. 💡
        """)

if __name__ == "__main__":
    main()
