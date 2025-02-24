Hi Team,

Task 2: Generative AI Application
Open WebUI → Streamlit for a more flexible UI.
Ollama → Langchain Ollama for improved LLM orchestration.
LangChain & DeepSeek-R1


Task 3: Implementing Naive RAG
Review and implement Naive RAG using the following resources:
Repo: Naive RAG → https://github.com/rizwan-ai/AgenticAI/tree/main/vanilla-rag
Notebook: Naive RAG Notebook → https://github.com/rizwan-ai/AgenticAI/blob/main/vanilla-rag/notebook.ipynb

Implementation Details:
LlamaIndex → LangChain integration.
Ollama → LangChain Ollama integration.
Replace sentence-transformers/all-MiniLM-L6-v2 with BAAI/bge-small-en-v1.5.
Migrate LlamaIndex VectorStoreIndex to Qdrant.
Use DeepSeek-R1:1.5b as the LLM.


Task 4: Document-based Chat using Naive RAG
Build FastAPI inference endpoints.
Deploy with Streamlit.


————————
What is Al Engineering?

1. Al Engineering feels closer to software engineering than to ML Engineering.

The term itself is very new, and "Al Engineering" evolved from ML Engineering. A big difference is that thanks to LLMs being easy to use (both via APIs, and locally) "Al Engineering" is much more about building a product first - and later on, getting around to tweaking the model itself.

2. To get good at Al Engineering, focus on the basics.
Understand what an LLM is (and how it works), how to evaluate them, how to use RAG, what Fine-tuning is, and how to optimize inference. All of these techniques are foundational, and will remain important in a few years' time as well.

3. Solving the problem is more important than using the latest Al tools.
Amusingly, a lot of teams miss this part: and they build overcomplicated Al solutions that do practically nothing for the business.

————————

RAG applications with advanced optimization techniques

1️⃣ Indexing
Indexing Optimization Techniques
- Data Pre-Processing
 -> Data Sources → Raw Data
 -> Data Extraction & Parsing → Parsed Data
 -> Data Cleaning & Noise Reduction → Cleansed Data
 -> Data Transformation → Transformed Data
- Chunking Strategies
 -> Fixed-size Chunking
 -> Recursive Chunking
 -> Document-based Chunking
 -> Semantic Chunking
 -> LLM-based Chunking
 -> Chunking → Data Chunks

2️⃣ Pre-Retrieval
Pre-Retrieval Optimization Techniques
- Query Transformation
 -> Query Rewriting
 => Raw Query → Query Re-writer (LLM) → Rewritten Query → Retriever → Retrieved Documents
 -> Query Expansion
 => Raw Query → Query Re-writer (LLM) → Expanded Query → Retriever → Retrieved Documents
- Query Decomposition
 -> Complex Input Query → Decomposition LLM → Sub-Queries → Retriever → Retrieved Documents → Sub-query Responses + Complex Input Query → LLM → Final Response
- Query Routing
 -> Retrieval Agent → Tools

3️⃣ Retrieval
Retrieval Optimization Strategies
- Metadata Filtering
 -> Query → Vector Database → Filtering Stage → Results
- Excluding Vector Search Outliers
 -> Top-k
 -> Distance Thresholding
 -> Autocut
- Hybrid Search
 -> Vector-based Semantic Search
 -> Keyword-based Search
- Embedding Model Fine-Tuning
 -> Pre-trained Embedding Model
 -> Fine-tuned Embedding Model

4️⃣ Post-Retrieval
Post-Retrieval Optimization Techniques
- Re-Ranking
 -> Retrieved Context → Reranker Model → Re-ranked Context
- Context Post-Processing
 -> Context Enhancement with Metadata → Sentence Window Retrieval
 -> Context Compression
 => Context → Compressor → Compressed Context
 => Embedding-based & Lexical-based Compression
- Prompt Engineering
 -> Chain of Thought (CoT)
 -> Tree of Thoughts (ToT)
 -> ReAct (Reasoning and Acting)
- LLM Fine-Tuning
 -> Pre-trained LLMs
 -> Domain-Specific Dataset → Fine-tuned LLM

⚡ I recommend implementing a validation pipeline to identify which parts of your RAG system need optimization and to assess the effectiveness of advanced techniques. Evaluating your RAG pipeline enables continuous monitoring and refinement, ensuring that optimizations positively impact retrieval quality and model performance!


### Commands: 
uvicorn task_4_fastapp:app --reload
docker run -p 6333:6333 qdrant/qdrant
streamlit run task_4_app_streamlit.py
