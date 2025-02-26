from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

llama = OllamaLLM(model="deepseek-r1:1.5b")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

input_dir_path = '/Users/usmanidrees/data'

loader = DirectoryLoader(input_dir_path, glob="**/*.pdf", recursive=True)
docs = loader.load()

for i, doc in enumerate(docs):
    print(f"Document {i}:")
    print(doc.page_content)
    print("=" * 50)

qdrant_client = QdrantClient(host="localhost", port=6333)

collection_name = "document_embeddings"
try:
    print(f"Deleting the existing collection `{collection_name}` if it exists...")
    qdrant_client.delete_collection(collection_name=collection_name)
    print(f"Collection `{collection_name}` deleted successfully.")
except Exception as e:
    print(f"Error while deleting the collection: {e}")

try:
    print(f"Creating a new collection `{collection_name}` with 384 dimensions...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Collection `{collection_name}` created successfully.")
except Exception as e:
    print(f"Error while managing collection: {e}")

document_store = {}
for i, doc in enumerate(docs):
    embedding = embed_model.embed_documents([doc.page_content])

    if isinstance(embedding[0], list):
        embedding = embedding[0]

    vector = embedding if isinstance(embedding, list) else embedding.tolist()

    document_store[str(i)] = doc

    point = PointStruct(id=i, vector=vector)

    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
        )
    except Exception as e:
        print(f"Failed to insert point for document {i}: {e}")

query_embedding = embed_model.embed_query("What exactly is DSPy?")

query = query_embedding

try:
    query_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=5
    )

    print("Raw query result:", query_result)

    if query_result:
        print("Query results:")
        retrieved_docs = []
        for result in query_result.points:
            doc_id = result.id
            similarity_score = result.score

            document = document_store.get(str(doc_id))

            if document is None:
                print(f"Document with id {doc_id} not found in the store.")
                continue

            retrieved_docs.append(document.page_content)
            print(f"Document ID: {doc_id}")
            print(f"Document Text: {document.page_content}")
            print(f"Similarity Score: {similarity_score}")
            print("=" * 50)

        prompt_template = "Question: {question}\nContext:\n{context}\nAnswer:"
        prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

        chain = LLMChain(llm=llama, prompt=prompt)
        context = "\n".join(retrieved_docs)
        query_text = "What exactly is DSPy?"

        response = chain.run(question=query_text, context=context)

        print("Generated Answer:")
        print(response)

    else:
        print("No results found in the query.")

except Exception as e:
    print(f"Error during query execution: {e}")
