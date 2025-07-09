import os
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model = 'gpt-4o-mini')
# 
# llm = GoogleGemini(api_key='AIzaSyAhUeySoblg5TFG37IHXuz_6mruby5GwvQ',temperature=0.4)

file_path = r"data.csv"
df = pd.read_csv(file_path)
sm = SmartDataframe(file_path,config={'llm':llm,"custom_whitelisted_dependencies": ["scikit-learn"]})

# response = sm.chat('give me the product distribution among sales')
agent_executor = create_pandas_dataframe_agent(
    llm,
    pd.read_csv(file_path),
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    allow_dangerous_code=True
)

# temp_path = r'C:\Users\navab\OneDrive\Desktop\Live_in_lab_V\exports\charts\temp_chart.png'
# if os.path.exists(temp_path):
#     time.sleep(4)
#     os.remove(temp_path)
# else:
#     print(response)