from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore  
from qdrant_client import QdrantClient
from langchain_ollama import OllamaLLM  
import PyPDF2
import os

app = FastAPI()
llm = OllamaLLM(model="deepseek-r1:1.5b")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(host="localhost", port=6333)  # Initialize the Qdrant client
qdrant_client = QdrantVectorStore(client=client, collection_name="document_embeddings", embedding=embed_model)


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


def load_pdfs_from_directory(directory: str):
    pdf_docs = []
    for root, dirs, files in os.walk(directory):  
        for file in files:
            if file.endswith(".pdf"):  
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                if text:
                    pdf_docs.append({"text": text, "metadata": {"source": file_path}})
    return pdf_docs


input_dir_path = '/Users/usmanidrees/data'  
pdf_docs = load_pdfs_from_directory(input_dir_path)
for i, doc in enumerate(pdf_docs):
    embedding = embed_model.embed_query(doc['text'])  
    
    try:
        qdrant_client.add_texts([doc['text']], metadatas=[{"id": i}])  
    except Exception as e:
        print(f"Failed to insert point for document {i}: {e}")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI app!"}

@app.post("/query")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
