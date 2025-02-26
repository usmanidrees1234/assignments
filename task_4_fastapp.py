import os
import shutil
import PyPDF2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore  
from qdrant_client import QdrantClient
from langchain_ollama import OllamaLLM  


# Initialize FastAPI app
app = FastAPI()

# Define directories
PRELOADED_DIRECTORY = "/Users/usmanidrees/data"  # Existing documents
UPLOAD_DIRECTORY = "/Users/usmanidrees/uploaded_pdfs"  # Uploaded documents

# Ensure both directories exist
os.makedirs(PRELOADED_DIRECTORY, exist_ok=True)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Initialize models and vector store
llm = OllamaLLM(model="deepseek-r1:1.5b")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(host="localhost", port=6333)
qdrant_client = QdrantVectorStore(client=client, collection_name="document_embeddings", embedding=embed_model)

# Function to extract text from a PDF
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text.strip()

# Function to process and store PDFs in Qdrant
def process_pdf(file_path: str, file_name: str):
    text = extract_text_from_pdf(file_path)
    
    if text:
        embedding = embed_model.embed_query(text)
        try:
            qdrant_client.add_texts([text], metadatas=[{"source": file_name}])
            return True
        except Exception as e:
            print(f"Failed to insert document '{file_name}' into Qdrant: {e}")
            return False
    return False

# Function to process all PDFs in a directory
def load_pdfs_from_directory(directory: str):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                process_pdf(file_path, file)

# Load and process pre-existing PDFs
load_pdfs_from_directory(PRELOADED_DIRECTORY)

# API Endpoint: Upload PDF file
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process and store the PDF
        success = process_pdf(file_path, file.filename)

        if success:
            return JSONResponse(content={"message": "File uploaded and indexed successfully."})
        else:
            return JSONResponse(content={"message": "File uploaded, but indexing failed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# API Endpoint: Query documents
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_documents(request: QueryRequest):
    query_text = request.query
    query_embedding = embed_model.embed_query(query_text)

    try:
        docs = qdrant_client.similarity_search(query_text, k=5)

        if docs:
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Question: {query_text}\nContext:\n{context}\nAnswer:"
            response = llm(prompt)

            return {"answer": response}
        else:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query execution: {str(e)}")

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI PDF processing app!"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

