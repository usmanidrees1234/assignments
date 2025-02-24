import llama_index
from importlib.metadata import version
print(f"LlamaIndex version: {version('llama_index')}")

# Configure embedding and LLM models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

# Update the embedding model to use BAAI/bge-small-en-v1.5
Settings.llm = Ollama(model="deepseek-r1:1.5b", request_timeout=300.0)  # Increased timeout to 5 minutes
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)

# Load data from the directory, filter for PDFs
from llama_index.core import SimpleDirectoryReader

input_dir_path = '/Users/usmanidrees/data'  # Adjust this to the correct path where your PDFs are stored

# Load PDF documents
loader = SimpleDirectoryReader(
    input_dir=input_dir_path,
    required_exts=[".pdf"],  # Filtering for .pdf files
    recursive=True
)
docs = loader.load_data()

# Print the documents (for debugging purposes)
for i, doc in enumerate(docs):
    print(f"Document {i}:")
    print(doc.text)  # This will print the text of each document
    print("=" * 50)  # A separator for clarity

# Set up Qdrant connection
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize Qdrant Client
qdrant_client = QdrantClient(host="localhost", port=6333)  # Adjust this for your Qdrant instance

# Force deletion of the collection if it exists
collection_name = "document_embeddings"
try:
    # Delete the collection if it exists
    print(f"Deleting the existing collection `{collection_name}` if it exists...")
    qdrant_client.delete_collection(collection_name=collection_name)  # Force delete the collection
    print(f"Collection `{collection_name}` deleted successfully.")
except Exception as e:
    print(f"Error while deleting the collection: {e}")

# Create the collection again with correct dimension size of 384
try:
    print(f"Creating a new collection `{collection_name}` with 384 dimensions...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Ensure correct vector size of 384
    )
    print(f"Collection `{collection_name}` created successfully.")
except Exception as e:
    print(f"Error while managing collection: {e}")

# Embed documents and store them in Qdrant
document_store = {}  # To store documents by their doc_id for later retrieval
for i, doc in enumerate(docs):
    # Embed the document text using the '_embed' method
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

# Querying the Qdrant Vector Store
query_embedding = Settings.embed_model._embed("What exactly is DSPy?")  # Example query

# Construct the query object
query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5, mode=VectorStoreQueryMode.DEFAULT)

# Query Qdrant for the most similar documents using query_points (replaces deprecated search)
try:
    query_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,  # Correct argument name is 'query'
        limit=5  # Top 5 results
    )

    # Print the raw query result
    print("Raw query result:", query_result)

    # If query is successful, process the result
    if query_result:
        print("Query results:")
        retrieved_docs = []
        for result in query_result.points:  # Correct access to points in QueryResponse
            # Extracting `id` and `score` from ScoredPoint object
            doc_id = result.id
            similarity_score = result.score

            # Check if the document id exists in the document store
            document = document_store.get(str(doc_id))  # Retrieve the full document by its ID
            
            # If the document doesn't exist in the store, skip
            if document is None:
                print(f"Document with id {doc_id} not found in the store.")
                continue
            
            retrieved_docs.append(document.text)  # Collect the retrieved documents for context
            print(f"Document ID: {doc_id}")
            print(f"Document Text: {document.text}")  # Print the full document text
            print(f"Similarity Score: {similarity_score}")
            print("=" * 50)

        # Combine the query with the retrieved documents as context
        context = "\n".join(retrieved_docs)
        query_text = "What exactly is DSPy?"

        # Generating the response using a language model (e.g., GPT or Ollama)
        response = Settings.llm.complete(
            prompt=f"Question: {query_text}\nContext:\n{context}\nAnswer:",
            max_tokens=100,  # Adjust max_tokens based on your needs (lower if needed)
            request_timeout=300  # Increased timeout to 5 minutes
        )

        # Print the generated answer
        print("Generated Answer:")
        print(response)

    else:
        print("No results found in the query.")

except Exception as e:
    print(f"Error during query execution: {e}")
