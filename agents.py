from pydantic import BaseModel, Field
from typing import TypedDict, Optional, Annotated, List
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import Tool
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from plotly.graph_objs import Figure
import sqlite3
import sys
import io
import traceback
import subprocess
from contextlib import redirect_stdout, redirect_stderr
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

class QueryCorrecter(BaseModel):
    query: str
class Preprocess_Input(BaseModel):
    sql_task:str =  Field(None,description='sql task in the user asked input')
    vis_task:Optional[str] = Field(None,description='visualization task in the user input')

class QueryOutput(BaseModel):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

class VisualizOutput(BaseModel):
    code: str = Field(...,description='Python Code for the Visualization task')

class SQL_State(TypedDict):
    question: Annotated[List,add_messages]
    sql_task: str
    vis_task: str
    query: str
    code: str
    figure: Figure
    sql_error: str
    py_error: str
    sq_result: str
    answer: str


class Agent:

    def __init__(self, db_url, db_path):
        self.OPENAI = init_chat_model(model='gpt-4o', model_provider='openai')
        self.LLAMA = init_chat_model(model='llama-3.3-70b-versatile', model_provider='groq')
        self.db = SQLDatabase.from_uri(db_url)
        self.conn = sqlite3.connect(db_path)
        print('DB connected SUCCESSFULLY')
        print(self.db.dialect)
        print(self.db.get_usable_table_names())

    def choose_llm(self, model):
        self.llm = self.OPENAI if model == 'openai' else self.LLAMA

    def execute_python_code(self, code):
        PACKAGE_MAPPING = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "tf": "tensorflow"
        }
        output = io.StringIO()
        error = io.StringIO()

        try:
            with redirect_stdout(output), redirect_stderr(error):
                globals_dict = {}
                exec(code, globals_dict)
            return output.getvalue(), error.getvalue(), None, globals_dict.get('fig')
        except ImportError as e:
            package_name = str(e).split("'")[1]
            pip_package_name = PACKAGE_MAPPING.get(package_name, package_name)
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_package_name])
                return self.execute_python_code(code)
            except subprocess.CalledProcessError:
                return output.getvalue(), error.getvalue(), f"Failed to install package: {pip_package_name}"
        except Exception:
            return output.getvalue(), error.getvalue(), traceback.format_exc(), globals_dict.get('fig')

    def preprocess_input(self, state: SQL_State):
        preprocess_parser = PydanticOutputParser(pydantic_object=Preprocess_Input)
        preprocess_template = """ 
            ROLE : To Extract the SQL tasks and Visualizations tasks from The user Queries
            NOTE:
            IMPORTANT : generate vis task only if needed otherwise leave it None
            you should not generate any python codes as well
            try  to include visualization task as well
            you should not write the actual sql queries instead just give the question to write sql query
            generate the appropriate the visualization query to do the visualization task by further generating python code
            FORMAT_INSTRUCTION : {format_instructions}
            INPUT : {input}
              """
        preprocess_prompt = PromptTemplate(
            template=preprocess_template,
            input_variables=["input"],
            partial_variables={'format_instructions': preprocess_parser.get_format_instructions()}
        )
        preprocess_llm = preprocess_prompt | self.llm
        output = preprocess_llm.invoke({'input': state['question']})
        result = preprocess_parser.invoke(output).model_dump()
        return result

    def generate_query(self, state: SQL_State):
        sql_llm = self.llm.with_structured_output(QueryOutput)

        query_prompt_template = """ Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

        Only use the following tables:
        you should only use the  columns present in the table_info don't create errors such as column not found
        {table_info}

        Question: {input} 

        Error : {error} // ignore if none
        """
        query_prompt = PromptTemplate.from_template(query_prompt_template)
        prompt = query_prompt.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 10,
                "table_info": self.db.get_table_info(),
                "input": state['sql_task'],
                'error': state.get('sql_error', None)
            }
        )
        return {'query': sql_llm.invoke(prompt).query}

    def execute_query(self, state: SQL_State):
        query_executor = QuerySQLDatabaseTool(db=self.db)
        query = state['query']
        df  = pd.read_csv('data.csv')
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_str = buffer.getvalue()
        # st_llm = self.llm.with_structured_output(QueryCorrecter)
        pd_llm = self.llm.with_structured_output(QueryCorrecter)
        pd_prompt_template = """ 
        Given SQL_QUERY for the database 
        INFO:
        dialect : {dialect}
        top_k : {top_k}
        table_info : {table_info}
        query: {input}
        info_data : {info_data}
        i want to make the query ,suits for running in pandas read_sql_query
        kindly correct the sql_query so it may be run in pandas database engine
        Columns : {columns} dont include unnamed columns since it is part of pandas df 
        """
        pd_prompt = PromptTemplate.from_template(pd_prompt_template)
        prompt = pd_prompt.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 10,
                "table_info": self.db.get_table_info(),
                "input": state['query'],
                'info_data':df_info_str,
                "columns":df.columns
                # 'error': state.get('sql_error', None)
            }
        )
        query_pd = pd_llm.invoke(prompt).query
        df = pd.read_sql_query(query_pd, self.conn)
        result = query_executor.invoke(query)
        df.to_csv('data.csv')
        error = None
        if 'error' in result:
            error = result
        return {'sq_result': result, 'sql_error': error}

    def generate_answer(self, state: SQL_State):
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n"
            "Your answer should be more humanly explaining the data\n"
            f'Question: {state["sql_task"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["sq_result"]}'
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    def generate_code(self, state: SQL_State):
        df = pd.read_sql_query(state['query'], self.conn)
        df.to_csv('data.csv')
        df.to_csv('data.csv')
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_str = buffer.getvalue()
        visualize_template = """ 
            ROLE : To Generate the Python Code For the visualization query asked to DO!
            NOTE:
            YOU SHOULD ONLY USE the DATA in 'data.csv' , dont create your own data
            {info_data}
            IMPORTANT : Try to Generate the code error freely
            It is Encourages to use PLOTLY instead MATPLOTLIB
            Always define the function(def plot()) with returns of fig
            SHOULD CALL THE FUNCTION BY
            fig = plot()
            AVOID fig.show()
            INPUT : {input}
            SAMPLE_DATA : {data}
              """
        visualize_prompt = PromptTemplate(
            template=visualize_template,
            input_variables=["input", "data"],
        )
        visualize_llm = visualize_prompt | self.llm.with_structured_output(VisualizOutput)
        output = visualize_llm.invoke({'input': state['vis_task'], 'data': df.head(100),'info_data':df_info_str})
        code = output.code
        return {'code': code}

    def execute_code(self, state: SQL_State):
        out, err, trace, fig = self.execute_python_code(state['code'])
        return {'py_error': trace,'figure':fig}

    def build_graph(self):
        Graph = StateGraph(SQL_State)

        # Add nodes
        Graph.add_node('preprocess_input', self.preprocess_input)
        Graph.add_node('generate_query', self.generate_query)
        Graph.add_node('execute_query', self.execute_query)
        Graph.add_node('generate_answer', self.generate_answer)
        Graph.add_node('generate_code', self.generate_code)
        Graph.add_node('execute_code', self.execute_code)

        # Add edges
        Graph.add_edge(START, 'preprocess_input')
        Graph.add_edge('preprocess_input', 'generate_query')
        Graph.add_edge('generate_query', 'execute_query')

        # Conditional edge after execute_query
        Graph.add_conditional_edges(
            'execute_query',
            lambda SQL_State: 'generate_answer' if SQL_State.get('sql_error', None) is None else 'generate_query'
        )

        # Conditional edge after generate_answer
        Graph.add_conditional_edges(
            'generate_answer',
            lambda SQL_State: END if SQL_State.get('vis_task', None) is None else 'generate_code'
        )

        # Edge from generate_code to execute_code
        Graph.add_edge('generate_code', 'execute_code')

        # Conditional edge after execute_code
        Graph.add_conditional_edges(
            'execute_code',
            lambda SQL_State: END if SQL_State.get('py_error', None) is None else 'generate_code'
        )

        checkpointer = MemorySaver()
        graph = Graph.compile()

        return graph
    

def main():
    # Initialize the Agent with a SQLite database URL
    db_url = "sqlite:///Chinook.db"  # Replace with your actual SQLite database path
    db_path = db_url[db_url.rfind("/")+1:]
    print(db_path)
    agent = Agent(db_url,db_path)

    # Choose the LLM (e.g., 'openai' or 'groq')
    agent.choose_llm(model='openai')

    # Define the user's question
    user_question = "Show me the total sales by region and visualize it as a bar chart."

    # Initialize the state with the user's question
    config = {
"configurable": {
"thread_id": 1,
"checkpoint_ns": "namespace-1",
"checkpoint_id": "checkpoint-1"
}
    }
    # Build the graph
    graph = agent.build_graph()

    # Run the graph with the initial state
    final_state = graph.invoke({'question':'How many Invoices were there in 2009 and 2011? What are the respective total sales for each of those years?'},config=config)

    # Print the final state to see the results
    print("Final State:")
    print(final_state)

    # Print the answer and visualization code (if any)
    if final_state.get('answer'):
        print("\nAnswer:")
        print(final_state['answer'])

    if final_state.get('code'):
        print("\nVisualization Code:")
        print(final_state['code'])

    if final_state.get('py_error'):
        print("\nPython Execution Error:")
        print(final_state['py_error'])

    if final_state.get('sql_error'):
        print("\nSQL Execution Error:")
        print(final_state['sql_error'])


# if __name__ == "__main__":
#     main()