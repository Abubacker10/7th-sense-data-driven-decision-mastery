import streamlit as st
import pandas as pd
# import pyodbc
# import json
import asyncio
# from lightning_rag import rag
from analytic_summary import SummaryGenerator
from QA_agent import agent_executor,sm
from lida_agent import do_lida
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import subprocess
# from Rag import rag
import time
# import matplotlib.backends.backend_tkagg
# matplotlib.use('tkagg')
from langchain_groq.chat_models import ChatGroq
# from streamlit_pandas_profiling import st_profile_report
# from pandas_profiling import ProfileReport
import sqlite3
from agents import Agent

input = None
report = None 

# llm = ChatGroq(model_name="llama-3.1-8b-instant")
async def run_summary_generation():
    generator = SummaryGenerator()
    
    if not generator.load_data("data_results.txt"):
        return
        
    generator.initialize_chains()
    
    try:
        summary = await generator.generate_report()
        print("Final Summary:")
        # print(summary)
        global report
        report = summary
        print('REPORT')
        print(report)
        return summary
    except Exception as e:
        print(f"Error generating report: {e}")


# Run the async function and get the output

def run_script():
    with open('Data_Analysis.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})

    # global report
    

# Function to convert CSV to SQLite
def csv_to_sqlite(df, db_path, table_name="data"):
    # df = pd.read_csv(csv_file)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return db_path

# Initialize session state for chatbot
if "db_path" not in st.session_state:
    st.session_state.db_path = None
if "messages" not in st.session_state:
    st.session_state.messages = []
    
t = """
   Hey now you are a data analyst you are given with the abstracted data details which contains many groups of data , it seggreates and aggregates the data iinformations as weel
   it have some outlier detection as well and it has the whole the descriptive statistics of the data with normal distribution test with some EDA
   the role you have to play is data analysis you should do descriptive analysis to help the retail business growth in the competitive market 
   YOU ARE DATA ANALYST your results should be more easier or laymen to understand
   provide like a report of analysis"""

con = 'You are a Data analyst you should provide the response as it was done by human'

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["File Upload","Analytics Report","EDA","Quick Insights","Chatbot"])

if page=='EDA':
    if 'activate_chat' not in st.session_state:
        st.session_state['activate_chat'] = True
        st.session_state.messages = []

    if st.session_state.activate_chat == True:
            
            if prompt := st.chat_input("Ask your question ?"):
                with st.chat_message("user", avatar = '👨🏻'):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", 
                                                "avatar" :'👨🏻',
                                                "content": prompt})
                result = sm.chat(prompt)
                # result = .invoke(prompt)['output']
                temp_path = r'exports\charts\temp_chart.png'
                if os.path.exists(temp_path):
                    st.image(temp_path)
                    time.sleep(4)
                    os.remove(temp_path)
                else:
                    with st.chat_message("assistant", avatar='🤖'):
                        st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", 
                                                    "avatar" :'🤖',
                                                    "content": result})
            else:
                st.markdown(
                    'Connect with your data to chat'
                    )

elif page == "File Upload":
    input = "File Upload"
    st.subheader("Upload your Data File")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file:
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)
        
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type!")
                df = None
            
            if df is not None:
                st.write("Preview of the uploaded file:")
                st.dataframe(df.head())
                if os.path.exists('data.csv'):
                    os.remove('data.csv')
                if os.path.exists('data_results.txt'):
                    os.remove('data_results.txt')
                df.to_csv('data.csv')
                with st.spinner("Executing the notebook... Please wait."):
                    run_script()
                    # os.system("python Data_Analysis.py")
                    # time.sleep(4)
                    os.system("python lightning_rag.py") 
                    # subprocess.run(["python", "lida_agent.py"])
                    st.success('Completed') 
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif page=='Q/A':
    if 'activate_chat' not in st.session_state:
        st.session_state['activate_chat'] = True
        st.session_state.messages = []

    if st.session_state.activate_chat == True:
            # st.write("""Sample queries
            #             What was the best month for sales? How much was earned that month?
            #             What city sold the most product?
            #             What time should we display advertisements to maximize the likelihood of customers buying products?
            #             What products are most often sold together?
            #             What product sold the most? Why do you think it sold the most?""")
            if prompt := st.chat_input("Ask your question ?"):
                with st.chat_message("user", avatar = '👨🏻'):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", 
                                                "avatar" :'👨🏻',
                                                "content": prompt})
                # result = sm.chat(prompt)
                result = agent_executor.invoke(prompt)['output']
                temp_path = r'exports\charts\temp_chart.png'
                if os.path.exists(temp_path):
                    st.image(temp_path)
                    time.sleep(4)
                    os.remove(temp_path)
                else:
                    with st.chat_message("assistant", avatar='🤖'):
                        st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", 
                                                    "avatar" :'🤖',
                                                    "content": result})
            else:
                st.markdown(
                    'Ask your Queris'
                    )
elif page=='Analytics Report':
    st.subheader('Analysis results:')
    report = asyncio.run(run_summary_generation())
    st.write(report)
    
elif page=="Quick Insights":
    Visuals,Goals = do_lida()
    tab_names = [f"Goal {i+1}" for i in range(len(Goals))]
    tabs = st.tabs(tab_names)

    for i, tab in enumerate(tabs):
        with tab:
            st.header(tab_names[i])
            st.write(Goals[i].question)
            st.image(Visuals[i]) 

elif page == "Chatbot":
    st.title("📊 Data Question Answering Chatbot")
    
    # Sidebar for file upload or database connection
    # with st.sidebar:
    #     st.header("Upload Data")
    #     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="chatbot_uploader")
    if os.path.exists('data.csv')==False:
        st.header('Upload yur data first')
    else:
        uploaded_df = pd.read_csv('data.csv')
    if not uploaded_df.empty:
            db_path = "temp.db"
            db_uri = "sqlite:///temp.db"
            st.session_state.db_path = csv_to_sqlite(uploaded_df, db_path)
            st.success(f"CSV file uploaded and converted to SQLite database: {db_path}")

        
        
            db_uri = f"sqlite:///{db_path}"
            agent = Agent(db_uri, st.session_state.db_path)
            agent.choose_llm('openai')
            graph = agent.build_graph()
            st.session_state.graph = graph

    # Chatbot Interface
    st.header("Chatbot")

    # Display chat messages
    if prompt := st.chat_input("Ask your question ?"):
        with st.chat_message("user", avatar = '👨🏻'):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", 
                                        "avatar" :'👨🏻',
                                        "content": prompt})
        
        if 'graph' in st.session_state:
            result = st.session_state.graph.invoke({'question': prompt})
            
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(result['answer'])
            if result.get('figure',None):
                st.plotly_chart(result['figure'])
            st.session_state.messages.append({"role": "assistant", 
                                            "avatar" :'🤖',
                                            "content": result['answer']})
        else:
            with st.chat_message("assistant", avatar='🤖'):
                st.error("Please upload a CSV file or connect to a database first")
            st.session_state.messages.append({"role": "assistant", 
                                            "avatar" :'🤖',
                                            "content": "Please upload a CSV file or connect to a database first"})

elif page == "Logout":
    st.success("Logged out successfully.")
    # Clear any necessary session state variables
    st.session_state.clear()