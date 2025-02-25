from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore  # Correct class
from qdrant_client import QdrantClient
from langchain_ollama import OllamaLLM  # Correct import from langchain-ollama
import PyPDF2
import os

app = FastAPI()

# Initialize Ollama model (using the langchain-ollama package)
llm = OllamaLLM(model="deepseek-r1:1.5b")

# Initialize HuggingFaceEmbeddings (without trust_remote_code argument)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Set up Qdrant client
client = QdrantClient(host="localhost", port=6333)  # Initialize the Qdrant client

# Initialize Qdrant with the embeddings model, corrected usage
qdrant_client = QdrantVectorStore(client=client, collection_name="document_embeddings", embedding=embed_model)

# Define a function to extract text from PDF files
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

# Define a function to recursively load PDFs and extract text
def load_pdfs_from_directory(directory: str):
    pdf_docs = []
    for root, dirs, files in os.walk(directory):  # Recursively walk through the directory
        for file in files:
            if file.endswith(".pdf"):  # Filter for PDF files
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                if text:
                    pdf_docs.append({"text": text, "metadata": {"source": file_path}})
    return pdf_docs

# Load and process documents
input_dir_path = '/Users/usmanidrees/data'  # Adjust to the correct path where your PDFs are stored

# Load all PDF files using the recursive method
pdf_docs = load_pdfs_from_directory(input_dir_path)

# Insert embeddings into Qdrant
for i, doc in enumerate(pdf_docs):
    # Pass the text directly as a string to embed_query
    embedding = embed_model.embed_query(doc['text'])  # Correct method to get embeddings for text
    
    # Insert into Qdrant
    try:
        qdrant_client.add_texts([doc['text']], metadatas=[{"id": i}])  # Insert text directly into Qdrant
    except Exception as e:
        print(f"Failed to insert point for document {i}: {e}")

# Pydantic model to accept query input
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI app!"}

@app.post("/query")
async def query_documents(request: QueryRequest):
    query_text = request.query

    # Embed the query using the local Hugging Face model
    query_embedding = embed_model.embed_query(query_text)  # Pass the query text as a string directly
    
    # Use the vector store to search for similar documents
    try:
        # Query the Qdrant vector store
        docs = qdrant_client.similarity_search(query_text, k=5)  # Retrieve top 5 similar documents
        
        if docs:
            # Retrieve text and generate response
            context = "\n".join([doc.page_content for doc in docs])

            # Use Ollama model for text generation (you can replace it with GPT, etc.)
            prompt = f"Question: {query_text}\nContext:\n{context}\nAnswer:"
            response = llm(prompt)
            
            return {"answer": response}
        else:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query execution: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
