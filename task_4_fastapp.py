from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
import json

app = FastAPI()

# Set up the embedding and LLM models
Settings.llm = Ollama(model="deepseek-r1:1.5b", request_timeout=300.0)  # Increased timeout
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)

# Initialize Qdrant Client
qdrant_client = QdrantClient(host="localhost", port=6333)  # Adjust this for your Qdrant instance
collection_name = "document_embeddings"

# Load PDFs from the directory
input_dir_path = '/Users/usmanidrees/data'  # Adjust to the correct path where your PDFs are stored
loader = SimpleDirectoryReader(input_dir=input_dir_path, required_exts=[".pdf"], recursive=True)
docs = loader.load_data()

# Store documents and their embeddings in Qdrant
document_store = {}  # To store documents by their doc_id for later retrieval
for i, doc in enumerate(docs):
    # Embed the document text using the embedding model
    embedding = Settings.embed_model._embed(doc.text)
    
    # Store the document in the dictionary for easy retrieval later
    document_store[str(i)] = doc
    
    # Ensure that embeddings are correctly formatted as lists or arrays
    if isinstance(embedding, list) or isinstance(embedding, tuple):
        vector = embedding
    else:
        # Handle any other case (e.g., converting numpy arrays to lists)
        vector = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    
    # Create a PointStruct to insert the document embedding into Qdrant
    point = PointStruct(id=i, vector=vector)  # Directly use the list as the vector

    # Insert the document embedding and associated doc_id into Qdrant
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],  # Insert the PointStruct
        )
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
    query_embedding = Settings.embed_model._embed(query_text)

    # Construct the query object
    query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5, mode=VectorStoreQueryMode.DEFAULT)

    try:
        # Query Qdrant for the most similar documents
        query_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=5  # Top 5 results
        )

        if query_result and query_result.points:
            # Retrieve documents and generate response
            retrieved_docs = []
            for result in query_result.points:
                doc_id = result.id
                document = document_store.get(str(doc_id))  # Retrieve the document from the store
                if document is not None:
                    retrieved_docs.append(document.text)

            context = "\n".join(retrieved_docs)

            # Use Ollama model for text generation (you can replace it with GPT, etc.)
            response = Settings.llm.complete(
                prompt=f"Question: {query_text}\nContext:\n{context}\nAnswer:",
                max_tokens=100,
                request_timeout=300  # Increased timeout to 5 minutes
            )
            
            return {"answer": response}
        else:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query execution: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
