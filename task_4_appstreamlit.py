import streamlit as st
import requests
import time

FASTAPI_URL = "http://127.0.0.1:8000/query"
st.title("Document-based Chat using Naive RAG")


if 'history' not in st.session_state:
    st.session_state.history = []


user_query = st.text_input("Ask a question:")
answer_placeholder = st.empty()

def clean_answer(answer):
    """ Clean and format the raw answer from the API """
    cleaned_answer = answer.replace("<think>", "").replace("</think>", "").strip()
    formatted_answer = cleaned_answer.replace("\n", "<br>")
    formatted_answer = formatted_answer.replace("**Answer**:", "<strong>Answer:</strong>")
    return formatted_answer

if user_query:
    st.session_state.history.append(f"**User**: {user_query}")

    try:
        response = requests.post(FASTAPI_URL, json={"query": user_query}, stream=True)
        
        if response.status_code == 200:
            answer = ''
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    answer += chunk
                    formatted_answer = clean_answer(answer)
                    answer_placeholder.markdown(formatted_answer, unsafe_allow_html=True)
                    time.sleep(0.05)  
            st.session_state.history.append(f"**Answer**: {answer}")
        else:
            error_message = response.json().get("detail", "Unknown error")
            st.error(f"Error: {error_message}")
            st.session_state.history.append(f"**Error**: {error_message}")
    except Exception as e:
        st.error(f"Error contacting the API: {e}")
        st.session_state.history.append(f"**Error**: {e}")

st.subheader("Conversation History")
for entry in st.session_state.history:
    st.markdown(entry)
