import os
import requests
import streamlit as st
import nltk
from rake_nltk import Rake
from dotenv import load_dotenv

# ==========================
# NLTK SETUP
# ==========================
nltk.download("punkt")
nltk.download("stopwords")

# ==========================
# RAKE SETUP
# ==========================
rake = Rake()

def extract_keywords_rake(text, max_keywords=5):
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()
    return ranked_phrases[:max_keywords]


# ==========================
# LOAD ENV VARIABLES
# ==========================
load_dotenv()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "openrouter/free",
    "google/gemma-2b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-mini-4k-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]

LEVELS = ["Beginner", "Intermediate", "Advanced"]
MAX_HISTORY_MESSAGES = 20


# ==========================
# SYSTEM PROMPT
# ==========================
def build_system_prompt(level: str) -> str:
    return f"""
You are an AI tutor helping a {level} level learner.

Follow this response structure exactly:

1) Simple explanation (1 short paragraph)
2) Real-world example (1 short paragraph)
3) If necessary, provide simple mathematical explanation
4) Quiz 1:
Question?

A) option
B) option
C) option
D) option

5) Suggest next topic

Rules:
- Clear headings
- Beginner friendly
- No repeated ideas
- Do NOT give quiz answers
"""


# ==========================
# OPENROUTER CALL
# ==========================
def call_openrouter(api_key, level, history, user_message, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": build_system_prompt(level)}]
    messages.extend(history[-MAX_HISTORY_MESSAGES:])

    # 🔥 RAKE keyword extraction
    keywords = extract_keywords_rake(user_message)
    st.write("🔎 Extracted Keywords:", keywords)

    keyword_string = " | ".join(keywords)

    enhanced_user_message = f"""
User Question:
{user_message}

Important Keywords:
{keyword_string}

Focus strongly on these keywords while answering.
"""

    messages.append({"role": "user", "content": enhanced_user_message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"API Error: {response.text}")


# ==========================
# STREAMLIT APP
# ==========================
def main():
    st.set_page_config(page_title="AI Tutor", page_icon="🎓")
    st.title("🎓 AI Tutor")
    st.caption("Learn any topic with structured explanations, examples, math, and quizzes.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "level" not in st.session_state:
        st.session_state.level = "Beginner"
    if "model" not in st.session_state:
        st.session_state.model = FREE_MODELS[0]

    # Sidebar
    with st.sidebar:
        st.header("⚙ Settings")

        st.session_state.level = st.selectbox(
            "Choose Learning Level",
            LEVELS,
            index=LEVELS.index(st.session_state.level),
        )

        st.session_state.model = st.selectbox(
            "Choose Model",
            FREE_MODELS,
            index=FREE_MODELS.index(st.session_state.model),
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error("❌ OPENROUTER_API_KEY not found in .env file")
        st.stop()

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_prompt = st.chat_input("Ask me to teach any topic...")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    assistant_text = call_openrouter(
                        api_key,
                        st.session_state.level,
                        st.session_state.messages,
                        user_prompt,
                        st.session_state.model,
                    )

                    st.markdown(assistant_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )

                except Exception as e:
                    st.error(f"⚠️ {e}")


if __name__ == "__main__":
    main()
