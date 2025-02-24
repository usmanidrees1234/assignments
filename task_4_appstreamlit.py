import streamlit as st
import requests
import time

# FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/query"

# Streamlit UI
st.title("Document-based Chat using Naive RAG")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input for query
user_query = st.text_input("Ask a question:")

# Placeholder for the answer, which will update during streaming
answer_placeholder = st.empty()

def clean_answer(answer):
    """ Clean and format the raw answer from the API """
    # Remove any <think> tags or other special formatting
    cleaned_answer = answer.replace("<think>", "").replace("</think>", "").strip()
    
    # Format the answer to make it more readable (replace newlines with breaks)
    formatted_answer = cleaned_answer.replace("\n", "<br>")
    
    # Further improve formatting if needed (optional)
    formatted_answer = formatted_answer.replace("**Answer**:", "<strong>Answer:</strong>")
    
    return formatted_answer

if user_query:
    # Append user query to history
    st.session_state.history.append(f"**User**: {user_query}")
    
    # Send the query to the FastAPI inference endpoint
    try:
        response = requests.post(FASTAPI_URL, json={"query": user_query}, stream=True)
        
        if response.status_code == 200:
            answer = ''
            # Streaming the response character by character
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    answer += chunk
                    # Clean and format the answer
                    formatted_answer = clean_answer(answer)
                    answer_placeholder.markdown(formatted_answer, unsafe_allow_html=True)
                    time.sleep(0.05)  # Slight delay to simulate streaming effect
            st.session_state.history.append(f"**Answer**: {answer}")
        else:
            error_message = response.json().get("detail", "Unknown error")
            st.error(f"Error: {error_message}")
            st.session_state.history.append(f"**Error**: {error_message}")
    except Exception as e:
        st.error(f"Error contacting the API: {e}")
        st.session_state.history.append(f"**Error**: {e}")

# Display conversation history
st.subheader("Conversation History")
for entry in st.session_state.history:
    st.markdown(entry)
