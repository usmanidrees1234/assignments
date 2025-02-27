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























/* Output */
/*
    (.venv) usmanidrees@Macbooks-MacBook-Pro-4.local /Users/usmanidrees 
% python task3.py
Document 0:
DSPy - Programming—not prompting—LMs

DSPy is the framework for programming—rather than prompting—language models. It allows you to iterate fast on building modular AI systems and offers algorithms for optimizing their prompts and weights, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

DSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional Python code and use DSPy to teach your LM to deliver high-quality outputs.

Getting Started I: Install DSPy and set up your LM

> pip install -U dspy

Local LMs on your laptop

First, install Ollama and launch its server with your LM.

> curl -fsSL https://ollama.ai/install.sh | sh

> ollama run llama3.2:1b

Then, connect to it from your DSPy code.

import dspy

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')

dspy.configure(lm=lm)
==================================================
Deleting the existing collection `document_embeddings` if it exists...
Collection `document_embeddings` deleted successfully.
Creating a new collection `document_embeddings` with 384 dimensions...
Collection `document_embeddings` created successfully.
Document 0 embedding:
[-0.08103113621473312, -0.028127899393439293, -0.06736597418785095, -0.00989273190498352, 0.02372613176703453, -0.031538769602775574, 0.0130761181935668, -0.009126648306846619, 0.05502418056130409, -0.0108196921646595, 0.02877570502460003, -0.026406943798065186, 0.021159609779715538, 0.00951111689209938, -0.051905810832977295, -0.014793092384934425, -0.026628518477082253, -0.01965745911002159, -0.008779026567935944, 0.0369243361055851, 0.035566296428442, -0.010941531509160995, -0.025888876989483833, -0.023607639595866203, 0.01476752944290638, 0.04562004655599594, -0.009623123332858086, -0.009885681793093681, -0.06353851407766342, -0.17520786821842194, 0.03556201606988907, 0.004161716438829899, 0.07265537977218628, -0.05674435570836067, 0.0452442541718483, 0.04093131050467491, 0.05670677870512009, -0.00876170676201582, -0.04542798548936844, 0.014150900766253471, -0.027859214693307877, -0.05162179470062256, 0.004715173039585352, -0.006720934994518757, -0.05116460472345352, -0.06788461655378342, -0.0008602063753642142, 0.010778434574604034, -0.0359082855284214, -0.021717097610235214, 0.011844438500702381, -0.026833614334464073, -0.02033454179763794, -0.002294618636369705, 0.01736290566623211, 0.0014200732111930847, 0.047425854951143265, 0.03661516681313515, 0.011058565229177475, 0.01619153469800949, 0.0057730949483811855, -0.008934575133025646, -0.08701810985803604, 0.07000627368688583, 0.017147811129689217, 0.050746478140354156, -0.012313412502408028, -0.04223433881998062, 0.05863865837454796, 0.04120084270834923, -0.040832217782735825, 0.017004387453198433, -0.01724867708981037, 0.06518275290727615, 0.06452665477991104, 3.173425739078084e-06, -0.027708662673830986, -0.0058116610161960125, 0.029059866443276405, 0.02012658305466175, -0.029618632048368454, -0.039748527109622955, 0.01865696907043457, -0.05867390334606171, 0.002608879469335079, 0.050959717482328415, -0.057818904519081116, -0.0178963802754879, -0.031463947147130966, -0.02982935681939125, -0.006379569880664349, -0.001828996348194778, -0.02411527745425701, 0.05206139013171196, -0.005833575502038002, -0.02356421761214733, -0.01249510794878006, -0.023623701184988022, -0.06570335477590561, 0.354987233877182, -0.051942162215709686, -0.0026316968724131584, -0.041430793702602386, 0.006804191507399082, -0.026918834075331688, -0.03910133242607117, -0.02116170898079872, 0.007263174746185541, -0.03490781784057617, -0.013852174393832684, -0.017448920756578445, 0.01863744482398033, 0.020367929711937904, -0.05918888747692108, 0.0001557419018354267, -0.02961214818060398, -0.03827015683054924, 0.012005887925624847, 0.014615034684538841, 0.06905806809663773, -0.0169479101896286, -0.007454856764525175, -0.011257557198405266, -0.05523495003581047, 0.031866881996393204, 0.028570331633090973, 0.01037867646664381, 0.10101573914289474, 0.01961705833673477, -0.0042907195165753365, 0.022016018629074097, -0.05647438019514084, -0.08099881559610367, -0.0009265838307328522, 0.06251203268766403, 0.005323741119354963, 0.028605535626411438, -0.028315100818872452, -0.0039920927956700325, -0.024185573682188988, 0.021325476467609406, 0.02512265369296074, 0.005564676132053137, 0.031688664108514786, -0.02134603261947632, 0.06940772384405136, -0.07081741094589233, -0.07141196727752686, -0.05722944438457489, 0.01898747868835926, -0.011324072256684303, 0.03675401955842972, 0.02403402142226696, -0.05497711896896362, 0.07369152456521988, 0.06576717644929886, 0.08018932491540909, -0.013283796608448029, -0.029519716277718544, 0.022671526297926903, -0.06433995813131332, -0.06314439326524734, 0.0034286838490515947, 0.06162968650460243, -0.06368865817785263, -0.11934702843427658, -0.029275190085172653, 0.013935511000454426, -0.053616996854543686, -0.0461009182035923, 0.07777855545282364, 0.05349268391728401, -0.06437192112207413, -0.02888796478509903, -0.05438997969031334, 0.0032956639770418406, -0.043111976236104965, -0.003290755208581686, 0.024776635691523552, 0.05311306565999985, 0.0030110683292150497, -0.009828985668718815, 0.060734450817108154, -0.06078982725739479, -0.022402243688702583, -0.01444005873054266, -0.07010071724653244, -0.038917750120162964, 0.03461400046944618, -0.031008347868919373, -0.029982710257172585, 0.021235572174191475, 0.038335397839546204, -0.053320419043302536, -0.013578410260379314, 0.021216807886958122, 0.02751009166240692, 0.0018429586198180914, -0.0072609372437000275, -7.581989484606311e-05, -0.033665936440229416, 0.10440772026777267, 0.04421708360314369, 0.06428967416286469, 0.010678411461412907, -0.021971356123685837, 0.042700815945863724, -0.006830613594502211, 0.016920361667871475, 0.09647080302238464, -0.09592180699110031, -0.06828650087118149, 0.04786287620663643, -0.0034269210882484913, -0.10072138160467148, 0.056936148554086685, -0.023068880662322044, 0.004040926694869995, -0.0143106859177351, -0.01652681827545166, -0.018328098580241203, 0.031364910304546356, -0.01687679998576641, -0.31616780161857605, 0.04586384817957878, 0.0708528533577919, 0.014354176819324493, 0.016496391966938972, -0.0644269585609436, 0.03132997080683708, 0.029142389073967934, -0.021384581923484802, 0.05503643676638603, 0.057608380913734436, -0.04440787434577942, 0.045933399349451065, -0.0808778926730156, -0.007539532147347927, 0.05332941561937332, -0.015449076890945435, 0.02554504945874214, 0.0018457442056387663, 0.05408173054456711, -0.0017909473972395062, 0.01442328654229641, 0.0221574567258358, -0.07294715940952301, -0.017779242247343063, -0.02067500166594982, 0.11260776221752167, -0.011115017347037792, 0.06199309229850769, -0.022110065445303917, 0.011172320693731308, 0.04184386134147644, -0.04279852658510208, -0.11481203138828278, 0.03464620187878609, -0.015617483295500278, 0.01998148113489151, 0.02862154133617878, 0.03215920180082321, -0.005924995988607407, -0.008353711105883121, 0.02306903526186943, 0.004438477102667093, -0.03996364027261734, -0.03382826969027519, 0.016550425440073013, 0.030567210167646408, -0.059845421463251114, 0.01929188333451748, -0.028263216838240623, 0.00782159622758627, 0.02170417830348015, -0.006829456426203251, 0.061853233724832535, -0.0548127256333828, 0.020138252526521683, -0.10282435268163681, -0.013636534102261066, -0.06240343675017357, -0.0149031737819314, -0.004569419659674168, -0.02785814180970192, 0.008122815750539303, -0.07594506442546844, -0.013974854722619057, -0.08877632766962051, 0.06734777241945267, -0.007199014071375132, 0.06050407886505127, -0.0017426079139113426, 0.0452578067779541, 0.1035388633608818, 0.030931612476706505, 0.043232906609773636, 0.012349838390946388, 0.04867534711956978, 0.002653322648257017, 0.004272834863513708, -0.0250873863697052, -0.004318242892622948, 0.04682636633515358, 0.04296496510505676, 0.04824965074658394, 0.07659196853637695, 0.05966382473707199, -0.02580554410815239, 0.04359447956085205, 0.022117864340543747, 0.002046074951067567, 0.03933798521757126, -0.06951543688774109, 0.015155041590332985, 0.0517120361328125, 0.05388574302196503, 0.008153626695275307, 0.01435728557407856, -0.24596130847930908, 0.05829845368862152, -0.04850619286298752, 0.07306541502475739, 0.0035808428656309843, 0.07036872953176498, -0.035842977464199066, -0.04821080341935158, -0.04073901101946831, -0.005331833381205797, -0.04915083572268486, 0.02894951030611992, 0.043674904853105545, -0.026528647169470787, -0.006303632166236639, -0.03228907287120819, 0.12227241694927216, 0.046489864587783813, -0.012021197006106377, 0.017141807824373245, -0.003202220890671015, 0.03439471870660782, 0.12629081308841705, -0.011241171509027481, -0.024145839735865593, 0.01843404956161976, -0.008602884598076344, -0.03214503824710846, 0.08078689128160477, 0.013805944472551346, 0.03871884569525719, 0.05972730368375778, 0.08858788013458252, -0.0004194621869828552, -0.03167368099093437, -0.011643254198133945, -0.023342207074165344, 0.030923636630177498, 0.029590904712677002, 0.02606622502207756, -0.06649049371480942, 0.02271856926381588, -0.005656559020280838, 0.014109884388744831, 0.0033981280867010355, 0.011254437267780304, -0.004998250398784876, -0.07521121203899384, -0.013457505032420158, 0.03743788227438927, -0.008758646436035633, 0.0002729485568124801, 0.007756032515317202, 0.005246635992079973, -0.0022504599764943123, 0.030803440138697624, 0.022942278534173965, -0.007362933363765478, 0.013379400596022606, 0.017580006271600723, 0.04879222810268402, -0.06481809914112091, 0.10909168422222137, -0.024858148768544197, -0.03766521438956261]
==================================================
Raw query result: points=[ScoredPoint(id=0, version=0, score=0.7955605, payload={}, vector=None, shard_key=None, order_value=None)]
Query results:
Document ID: 0
Document Text: DSPy - Programming—not prompting—LMs

DSPy is the framework for programming—rather than prompting—language models. It allows you to iterate fast on building modular AI systems and offers algorithms for optimizing their prompts and weights, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

DSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional Python code and use DSPy to teach your LM to deliver high-quality outputs.

Getting Started I: Install DSPy and set up your LM

> pip install -U dspy

Local LMs on your laptop

First, install Ollama and launch its server with your LM.

> curl -fsSL https://ollama.ai/install.sh | sh

> ollama run llama3.2:1b

Then, connect to it from your DSPy code.

import dspy

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')

dspy.configure(lm=lm)
Similarity Score: 0.7955605
==================================================
/Users/usmanidrees/task3.py:105: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
  chain = LLMChain(llm=llama, prompt=prompt)
/Users/usmanidrees/task3.py:109: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
  response = chain.run(question=query_text, context=context) */
