import streamlit as st
import pandas as pd
import sqlite3
import imgkit
from io import StringIO
import os

from agents import Agent


# Initialize session state
if "db_path" not in st.session_state:
    st.session_state.db_path = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to convert CSV to SQLite
def csv_to_sqlite(csv_file, db_path, table_name="data"):
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return db_path

# Function to execute SQL query
def execute_query(db_path, query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return pd.DataFrame(result, columns=columns)
    except Exception as e:
        conn.close()
        return str(e)

# Function to generate SQL and Python code
def generate_code(question):
    # Example: Generate SQL and Python code based on the question
    sql_code = f"SELECT * FROM data WHERE ..."  # Replace with actual SQL generation logic
    python_code = f"import pandas as pd\nimport sqlite3\n\n# Connect to the database\nconn = sqlite3.connect('{st.session_state.db_path}')\n\n# Execute the query\ndf = pd.read_sql_query('''{sql_code}''', conn)\n\n# Display the result\nprint(df)"
    return sql_code, python_code

# Streamlit App
st.title("📊 Data Question Answering Chatbot")

# Sidebar for file upload or database connection
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        db_path = "temp.db"
        db_uri = "sqlite:///temp.db"
        st.session_state.db_path = csv_to_sqlite(uploaded_file, db_path)
        st.success(f"CSV file uploaded and converted to SQLite database: {db_path}")

    st.header("Connect to SQLite Database")
    db_path = st.text_input("Enter SQLite database path (e.g., 'data.db')")
    if db_path and os.path.exists(db_path):
        st.session_state.db_path = db_path
        st.success(f"Connected to SQLite database: {db_path}")
    db_path = "temp.db"
    db_uri = "sqlite:///temp.db"
    agent = Agent(db_uri,db_path)
    agent.choose_llm('openai')
    graph = agent.build_graph()

# Chatbot Interface
st.header("Chatbot")

# Display chat messages
if prompt := st.chat_input("Ask your question ?"):
    with st.chat_message("user", avatar = '👨🏻'):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", 
                                    "avatar" :'👨🏻',
                                    "content": prompt})
    result = graph.invoke({'question':prompt})

    with st.chat_message("assistant", avatar='🤖'):
        st.markdown(result['answer'])
    if result['figure']:
        st.plotly_chart(result['figure'])
    st.session_state.messages.append({"role": "assistant", 
                                        "avatar" :'🤖',
                                        "content": result['answer']})