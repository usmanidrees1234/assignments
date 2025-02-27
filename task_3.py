from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

import concurrent.futures
import time

llama = OllamaLLM(model="deepseek-r1:1.5b")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

input_dir_path = '/Users/usmanidrees/data'
loader = DirectoryLoader(input_dir_path, glob="**/*.pdf", recursive=True)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = []
for doc in docs:
    for chunk in text_splitter.split_text(doc.page_content):
        split_docs.append({"text": chunk, "metadata": doc.metadata})

qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "document_embeddings"

try:
    qdrant_client.delete_collection(collection_name=collection_name)
except Exception:
    pass

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

document_store = {}

def process_doc(i, doc):
    embedding = embed_model.embed_documents([doc["text"]])[0]
    point = PointStruct(
        id=i,
        vector=embedding,
        payload={"text": doc["text"], "metadata": doc["metadata"]}
    )
    document_store[str(i)] = doc
    return point

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor() as executor:
    points = list(executor.map(process_doc, range(len(split_docs)), split_docs))

qdrant_client.upsert(collection_name=collection_name, points=points)

query_text = "What exactly is DSPy?"
query_embedding = embed_model.embed_query(query_text)

search_results = qdrant_client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5
)

retrieved_docs = []
for result in search_results:
    doc_text = result.payload["text"]
    similarity_score = result.score
    retrieved_docs.append(doc_text)

if retrieved_docs:
    prompt_template = "Question: {question}\nContext:\n{context}\nAnswer:"
    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    chain = LLMChain(llm=llama, prompt=prompt)
    context = "\n".join(retrieved_docs)
    response = chain.invoke({"question": query_text, "context": context})
    print(response)
else:
    print("No relevant documents found.")








"""
(.venv) usmanidrees@Macbooks-MacBook-Pro-4.local /Users/usmanidrees 
% python task3.py
/Users/usmanidrees/task3.py:62: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
  search_results = qdrant_client.search(
/Users/usmanidrees/task3.py:77: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
  chain = LLMChain(llm=llama, prompt=prompt)
{'question': 'What exactly is DSPy?', 'context': "DSPy - Programming—not prompting—LMs\n\nDSPy is the framework for programming—rather than prompting—language models. It allows you to iterate fast on building modular AI systems and offers algorithms for optimizing their prompts and weights, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.\nDSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional Python code and use DSPy to teach your LM to deliver high-quality outputs.\n\nGetting Started I: Install DSPy and set up your LM\n\n> pip install -U dspy\n\nLocal LMs on your laptop\n\nFirst, install Ollama and launch its server with your LM.\n\n> curl -fsSL https://ollama.ai/install.sh | sh\n\n> ollama run llama3.2:1b\n\nThen, connect to it from your DSPy code.\n\nimport dspy\nimport dspy\n\nlm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')\n\ndspy.configure(lm=lm)", 'text': '<think>\nAlright, so I\'m trying to understand what DSPy is. From the context provided, it seems like it\'s a framework or tool that allows programming without having to prompt the language model directly. Instead of writing commands and letting the model respond, you\'re supposed to write Python code in a declarative way, which means more like sentences rather than explicit instructions.\n\nOkay, so let me break this down step by step. First, I see that it stands for Declarative Self-improving Python. That makes sense because it\'s about being declarative—expressing what you want to achieve without specifying every single detail in advance. It’s like giving directions but not listing each step.\n\nThe framework allows building modular AI systems, which means it helps make your models more complex and flexible by breaking them down into smaller, manageable parts. This is useful because it can handle intricate tasks that might be too simple for traditional programming approaches.\n\nIt also offers algorithms for optimizing prompts and weights. That part seems a bit confusing. Prompts are the initial instructions given to a model, right? So optimizing those could mean tweaking how you set them up to get better results. Similarly, "weights" might refer to adjusting certain parameters in your code or model configuration to improve performance.\n\nThe context mentions that you can use DSPy with various types of language models—like simple classifiers and sophisticated RAG pipelines (which I think stands for Retrieval-Augmented Generation) as well as agent loops. That covers a range of AI tasks, from straightforward classification problems to more complex decision-making processes that rely on retrieval mechanisms.\n\nThe Getting Started section provides instructions on how to set up DSPy. First, you need to install the package using pip:\n```\npip install -U dspy\n```\nThen, you need an LLM (Language Model) like Ollama or LLaMA, which are popular choices for such setups.\n\nConnecting to the LLM involves running a shell command with its API key. Here\'s what that step looks like:\n```\ncurl -fsSL https://ollama.ai/install.sh | sh\nollama run llama3.2:1b\n\nThen, you need to set up your local LLM using this API key by calling dspy.configure() in your Python code:\n```python\nimport dspy\nimport dspy.LM\n\nlm = dspy.LM(\'ollama_chat/llama3.2\', api_base=\'http://localhost:11434\', api_key=\'\')\ndspy.configure(lm=lm)\n```\nThis part is a bit tricky because it involves integrating the LLM with the DSPy framework.\n\nI\'m still not entirely clear on how exactly you integrate the LLM into your code, but from what I can gather, you first set up your local LLM and then configure it with the LPY (Language Model-Y) object. Then, you use DSPy to handle prompts and optimize the model’s performance.\n\nOne thing that stands out is the mention of "self-improving," which implies that as you iterate through different strategies and configurations in your code, the system itself can improve its performance based on feedback from the LLM. This seems like a key feature for making your models more adaptable and robust over time.\n\nThe section also gives some sample tasks where DSPy could be useful:\n- **Simple Classifiers**: Maybe using it to build models that classify data with high accuracy.\n- **RAG Pipelines**: Retrieval-Augmented Generation sounds like having the model not only generate responses but also retrieve information from external sources, which can enhance performance in certain domains.\n\nFor agent loops, this might involve creating systems where AI agents within them use DSPy to dynamically adjust their behavior based on inputs or feedback from the environment. This could lead to more adaptive and intelligent systems.\n\nOverall, it seems like a versatile framework that bridges declarative programming with language models, allowing for more dynamic and efficient AI development compared to traditional programming approaches.\n</think>\n\n**Final Answer:**\n\nDSPy is a powerful framework designed to enhance AI development through declarative programming without direct prompting. Here\'s a concise breakdown:\n\n1. **Concept**: It allows building complex systems by expressing tasks as sentences, enabling flexibility and modularity.\n\n2. **Task Types**:\n   - **Classifiers**: Simple classification problems.\n   - **RAG Pipelines**: Combines retrieval of external data with model generation for enhanced performance.\n   - **Agent Loops**: Dynamic AI systems adjusting behavior based on feedback.\n\n3. **Integration**: Uses a local LLM (like Ollama) to provide insights and optimize models. DSPy configures this by integrating declarative prompts into your code.\n\n4. **Self-Improving**: As iterations occur, the system adapts and improves its performance through continuous feedback from the LLM.\n\n5. **Getting Started**:\n   - Install `pip install -U dspy`.\n   - Use `ollama run llama3.2:1b` to set up your local LLM.\n   - Configure it with `dspy.configure(lm=lm)` in Python code.\n\nDSPy offers a dynamic approach, enhancing AI models by leveraging declarative prompts and self-improvement through iterative learning from external data.'}
"""
