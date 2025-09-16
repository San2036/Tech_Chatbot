import os
import random
import streamlit as st
import pandas as pd
from datetime import datetime
import google.generativeai as genai

# ------------------------------
# Streamlit page setup
# ------------------------------
st.set_page_config(page_title="🌊 Ocean Hazard Chatbot", page_icon="💻")

# ------------------------------
# Gemini setup
# ------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "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")  # or set in Streamlit secrets
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model
model = genai.GenerativeModel("google/gemini-2.5-flash-lite")

# ------------------------------
# Chatbot function
# ------------------------------
def chatbot(input_text: str) -> str:
    try:
        response = model.generate_content([
            {"role": "system", "parts": 
                "You are an Ocean Hazard Assistant 🌊. "
                "Provide safety alerts and information about tsunamis, storm surges, flooding, coastal damage, "
                "early warnings, evacuation plans, and general preparedness. "
                "Be clear and concise, and always prioritize safety."
            },
            {"role": "user", "parts": input_text}
        ])
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error fetching response from Gemini: {e}"

# ------------------------------
# Conversation logging
# ------------------------------
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

# ------------------------------
# Streamlit App UI
# ------------------------------
def main():
    st.title("💻 Ocean Hazard Chatbot (Gemini)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "clear_flag" not in st.session_state:
        st.session_state.clear_flag = False

    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=150)
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        if os.path.exists("tech_chat_log.csv"):
            os.remove("tech_chat_log.csv")
        st.session_state.clear_flag = True
        st.success("Chat history cleared!")

    # Navigation buttons
    st.subheader("🏠 Home Page")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Chat"):
            st.session_state.page = "chat"
    with col2:
        if st.button("🕘 Conversation History"):
            st.session_state.page = "history"
    with col3:
        if st.button("ℹ️ About"):
            st.session_state.page = "about"

    # Chat page
    if st.session_state.page == "chat":
        st.subheader("💬 Ask Your Ocean Hazard Questions")

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

    # History page
    elif st.session_state.page == "history":
        st.subheader("🕘 Past Conversations")
        if os.path.exists("tech_chat_log.csv"):
            chat_df = pd.read_csv("tech_chat_log.csv")
            search_term = st.text_input("Search conversation:")
            filtered_df = chat_df[chat_df.apply(
                lambda row: search_term.lower() in str(row["User Input"]).lower() or
                            search_term.lower() in str(row["Bot Response"]).lower(),
                axis=1
            )] if search_term else chat_df
            for _, row in filtered_df.iterrows():
                st.markdown(f"**User:** {row['User Input']}")
                st.markdown(f"**Bot:** {row['Bot Response']}")
                st.markdown("---")
        else:
            st.info("No past conversations found.")

    # About page
    elif st.session_state.page == "about":
        st.subheader("ℹ️ About This Chatbot")
        st.write("""
        🤖 This chatbot is powered by **Google Gemini**  
        🌊 It provides real-time answers about:
        - Tsunamis
        - High waves
        - Storm surges
        - Coastal flooding
        - Evacuation & safety tips  

        **Tech stack used:**
        - Streamlit (UI)
        - Gemini API (chat intelligence)
        - CSV logging for conversation history  

        💡 No JSON intents needed — answers are generated dynamically by AI.
        """)

# ------------------------------
if __name__ == "__main__":
    main()
