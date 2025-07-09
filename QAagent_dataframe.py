import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
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

tools = Tool(
        name="repl_tool",
        func=PythonREPL,
        description="This is a test tool"
    )


df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)


FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do, what action to take
Action: python_repl_ast
Action Input: the input to the action, never add backticks "`" around the action input and if there is any plot save the plot
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
agent = create_pandas_dataframe_agent(ChatGroq(temperature=0.4), df, verbose=True,allow_dangerous_code=True,agent_executor_kwargs={'format_instructions':FORMAT_INSTRUCTIONS})


print(agent.invoke("show me barchart of gender distribution"))