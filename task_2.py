import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time

# Define a prompt template
prompt = PromptTemplate(input_variables=["text"], template="Generate a response to this: {text}")

# Initialize the Ollama model with the available model
#llm = Ollama(model="deepseek-r1:latest")  # Using the available model
llm = Ollama(model="deepseek-r1:1.5b")

# Create a chain that connects the prompt and the LLM
chain = LLMChain(llm=llm, prompt=prompt)

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display title and description for UI
st.title("AI Chatbot")
st.markdown("This is a professional chat interface that uses Ollama LLM to generate responses.")

# Function to simulate streaming response word by word
def simulate_streaming_response(response):
    words = response.split()  # Split the response into words
    full_response = ""
    
    for word in words:
        full_response += word + " "  # Add each word to the full response
        st.write(full_response)  # Display the response so far
        time.sleep(0.5)  # Simulate typing delay between words

# Get input from the user
user_input = st.text_input("Enter your message:")

# If the user input is not empty, run the chain
if user_input:
    # Generate response based on the user's input
    response = chain.run(user_input)
    
    # Store the user input and response in the chat history
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    st.session_state.chat_history.append({"role": "ai", "text": response})

    # Display the chat history with a professional layout
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You**: {message['text']}")
        else:
            st.markdown(f"**AI**: {message['text']}")
    
    # Simulate streaming of the AI's response (word by word)
    with st.spinner("AI is typing..."):  # Show a loading spinner
        simulate_streaming_response(response)
