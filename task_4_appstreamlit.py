import streamlit as st
import requests
import os

# FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8000"  # Update if FastAPI runs on a different host

# Create Streamlit UI
st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

# Sidebar: Chat History and File Upload
st.sidebar.title("📁 Chat History & Uploads")

# Create an area to store uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)  # Ensure temp directory exists
    
    # Save file temporarily
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Send file to FastAPI backend
    with st.spinner("Uploading..."):
        files = {"file": open(file_path, "rb")}
        response = requests.post(f"{FASTAPI_URL}/upload/", files=files)

        if response.status_code == 200:
            st.session_state["uploaded_files"].append(uploaded_file.name)
            st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")
        else:
            st.sidebar.error("❌ Upload failed.")

# Display uploaded files in sidebar
if st.session_state["uploaded_files"]:
    st.sidebar.subheader("Uploaded PDFs:")
    for file_name in st.session_state["uploaded_files"]:
        st.sidebar.text(f"📄 {file_name}")

# Chat Interface
st.title("💬 PDF Chat Assistant")
st.write("Ask questions about your PDFs!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
query = st.chat_input("Ask a question...")
if query:
    # Add user query to chat history
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Send query to FastAPI for response
    with st.spinner("Thinking..."):
        response = requests.post(f"{FASTAPI_URL}/query/", json={"query": query})

        if response.status_code == 200:
            answer = response.json().get("answer", "No response received.")
        else:
            answer = "❌ Error processing query."

    # Add AI response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
