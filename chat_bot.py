import os
import json
import random
import re
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="ocean Hazard Chatbot", page_icon="🌊")


def preprocess(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    filtered = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)


file_path = "tech_intents.json"
intents = []
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if "intents" in data:
            intents = data["intents"]
except Exception as e:
    st.error(f"Error loading intents: {e}")


patterns, tags, processed_patterns = [], [], []
tag_to_responses = {}
dynamic_tags = set()

if intents:
    for intent in intents:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            processed_patterns.append(preprocess(pattern))
            tags.append(intent["tag"])

        tag_to_responses[intent["tag"]] = intent.get("responses", [])
        if intent.get("dynamic", False):
            dynamic_tags.add(intent["tag"])


vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(processed_patterns) if processed_patterns else None


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


def fetch_dynamic_response(query):
    try:
        search_query = "+".join(query.strip().split())
        url = f"https://www.google.com/search?q={search_query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        # Try span-based results first
        for span in soup.select('.aCOpRe span'):
            text = span.get_text().strip()
            if len(text.split()) >= 6:
                return text

        
        for div in soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd"):
            text = div.get_text().strip()
            if len(text.split()) >= 6:
                return text

        return "I searched online but couldn't find a clear solution. Try rephrasing or being more specific."
    except Exception as e:
        return f"Sorry, I couldn't fetch online results due to an error: {e}"


def chatbot(input_text):
    if x_train is None:
        return "Sorry, I can't process your query right now."

    input_vec = vectorizer.transform([preprocess(input_text)])
    similarity = cosine_similarity(input_vec, x_train).flatten()
    best_index = np.argmax(similarity)
    confidence = similarity[best_index]

    # Low confidence fallback: Try Google if available
    if confidence < 0.3:
        return fetch_dynamic_response(input_text)

    matched_tag = tags[best_index]
    responses = tag_to_responses.get(matched_tag, [])

    if matched_tag in dynamic_tags:
        return fetch_dynamic_response(input_text)
    elif responses:
        return random.choice(responses)

    return "I understand your question, but I don't have a good answer yet."


def main():
    st.title("🌊 Ocean Hazard chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "clear_flag" not in st.session_state:
        st.session_state.clear_flag = False

    
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=150)
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        if os.path.exists("tech_chat_log.csv"):
            os.remove("tech_chat_log.csv")
        st.session_state.clear_flag = True
        st.success("Chat history cleared!")

   
    st.subheader("🏠 Home Page")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Chat"):
            st.session_state.page = "chat"
    with col2:
        if st.button("🕘 Conversation History"):
            st.session_state.page = "history"
    with col3:
        if st.button("ℹ About"):
            st.session_state.page = "about"

 
    if st.session_state.page == "chat":
        st.subheader("💬 Ask Your Tech Questions")

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

  
    elif st.session_state.page == "history":
        st.subheader("🕘 Past Conversations")
        if os.path.exists("tech_chat_log.csv"):
            chat_df = pd.read_csv("tech_chat_log.csv")
            search_term = st.text_input("Search conversation:")
            filtered_df = chat_df[chat_df.apply(
                lambda row: search_term.lower() in str(row["User Input"]).lower() or
                            search_term.lower() in str(row["Bot Response"]).lower(),
                axis=1
            )]
            for _, row in filtered_df.iterrows():
                st.markdown(f"*User:* {row['User Input']}")
                st.markdown(f"*Bot:* {row['Bot Response']}")
                st.markdown("---")
        else:
            st.info("No past conversations found.")

   
    elif st.session_state.page == "about":
        st.subheader("ℹ About This Chatbot")
        st.write("""
        🤖 This is a smart tech support chatbot built using:
        - *Streamlit* for the UI  
        - *Scikit-learn* for TF-IDF based intent matching  
        - *Google Search Snippet Fetching* for dynamic questions  
        - *Local CSV logging* of conversations  
        - *JSON-based intent training* via tech_intents.json  
        
        💡 To update or add responses, edit tech_intents.json.
        """)

if __name__ == "__main__":
    main()

