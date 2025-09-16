import os
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

# ------------------------------
# Streamlit setup
# ------------------------------
st.set_page_config(page_title="🌊 Ocean Hazard Chatbot", page_icon="💻")

# ------------------------------
# OpenRouter setup
# ------------------------------
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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"
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# ------------------------------
# Chatbot function
# ------------------------------
def chatbot(input_text: str) -> str:
    try:
        payload = {
            "model": "google/gemini-flash-1.5",  # Gemini via OpenRouter
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an Ocean Hazard Assistant 🌊. "
                        "Provide safety alerts and information about tsunamis, storm surges, flooding, "
                        "coastal damage, early warnings, evacuation plans, and preparedness. "
                        "Be clear, concise, and prioritize safety."
                    )
                },
                {"role": "user", "content": input_text}
            ]
        }
        response = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ Error: {data}"
    except Exception as e:
        return f"⚠️ Error fetching response from OpenRouter: {e}"

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
# Streamlit UI
# ------------------------------
def main():
    st.title("💻 Ocean Hazard Chatbot (Gemini via OpenRouter)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=150)
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        if os.path.exists("tech_chat_log.csv"):
            os.remove("tech_chat_log.csv")
        st.success("Chat history cleared!")

    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Chat"):
            st.session_state.page = "chat"
    with col2:
        if st.button("🕘 History"):
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
        🤖 Powered by **OpenRouter / Gemini 1.5 Flash**  
        🌊 Real-time answers on tsunamis, storm surges, flooding, coastal damage, and evacuation tips.  
        **Tech stack:** Streamlit + OpenRouter API + CSV logging
        """)

# ------------------------------
if __name__ == "__main__":
    main()
