import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import JSONLoader,CSVLoader
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate

# def rag():

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    )   

loader = CSVLoader(r"data_results.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents)

system = ("You are a data scientist and you have to analyse the data provided to you"
        "carefully look into every insights from the data"
        "Generate a well defined report of the data"
        """ 
we want the following sections in the result 
                                    1.Data Understanding
                                    2.What was in The data ?
                                    3.How and Why it was happened ?
                                    4.Beautiful insights
                                    5.Takeaways
                                    6.Any suggestions
                                    7.Future prespectives

                                    Kindly do the  favor of making the data valuable by taking away the values """)
human = """kindly analyse the data and response according to the user
            mostly you have to generate a precise analytical report
            question: {question}
            context: {context}
            answer: """
prompt = ChatPromptTemplate([
    ("system", system),
    ("human", human),
])

chain = prompt | _ | llm

print(chain.invoke({'question':'hi'}).content)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}



def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

print('3')
# # Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

# return graph

# 
# print(graph.invoke({'question':'analyse the data'}))