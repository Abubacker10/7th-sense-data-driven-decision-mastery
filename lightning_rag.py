from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# from langchain_ollama import ChatOllama

# model = ChatOllama(
#     model="llama3.1",
#     temperature=0.4,
#     # other params...
# )


from langchain_google_genai import GoogleGenerativeAIEmbeddings

# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key='AIzaSyAhUeySoblg5TFG37IHXuz_6mruby5GwvQ')
embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    # encode_kwargs = {'precision': 'binary'}
    )
# db_file_path = r"C:\Users\navab\OneDrive\Desktop\Live_in_lab_V\vector_stores\one-bit"

def rag():
    # if os.path.exists(db_file_path):
        # one_bit_vectorstore = FAISS.load_local(db_file_path, embeddings,allow_dangerous_deserialization=True)
    # else:
    loader = CSVLoader(r"data_results.txt")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    # print(docs)
    
    one_bit_vectorstore = FAISS.from_documents(documents, embeddings)
    # one_bit_vectorstore.save_local(db_file_path)
        
    one_bit_retriever = one_bit_vectorstore.as_retriever(
                                                        search_kwargs={"k": 3}
                                                        )

    contextualize_q_system_prompt  = """  You are an AI assistant helping the client/user to understand their data and make some immediate actions
                                            you have tasks like analysing the data they provided
                                            make sure that your analytical report should be understand by the laymen business man
                                            do intrepret everything provided to you

    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            # MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    model = ChatGroq(model_name="llama-3.1-8b-instant")
    # document_chain = create_stuff_documents_chain(model, prompt)
    # one_bit_retrieval_chain = create_retrieval_chain(retriever = one_bit_retriever, document_chain)
    history_aware_retriever = create_history_aware_retriever(
        model, one_bit_retriever, contextualize_q_prompt
    )
    system_prompt = (
        "You are a data scientist. "
        "analyse and intrepret the data"
        "try to answer within 2 lines"
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "perform end to end analysis "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
           
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # response = one_bit_retrieval_chain.invoke({"input": "What does the paper introduce?" })
    # print(response["answer"])
    # chat_history = []
    # question = 'i want to book ticket'
    # response = rag_chain.invoke({'input':question, "chat_history": chat_history})
    # chat_history.extend(
    #     [
    #         HumanMessage(content=question),
    #         AIMessage(content=response["answer"]),
    #     ]
    # )
    # print(response['answer'])
    return rag_chain
