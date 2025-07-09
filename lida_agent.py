import pandas as pd
from lida import Manager, TextGenerationConfig , llm  
from PIL import Image
from io import BytesIO
import base64
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

lida = Manager(text_gen = llm(provider="openai"))
textgen_config = TextGenerationConfig(n=1, 
                                        temperature=0.5, 
                                    
                                        use_cache=True)
def do_lida():
    summary = lida.summarize(r"data.csv")  
    # print(summary)
    goals = lida.goals(summary=summary, n=10)

    i = 0
    library = "seaborn"
    Visuals = []
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
    # Create the corresponding data visualization for each goal
    # print(pd.DataFrame(goals))
    Goals = []
    for i in range(len(goals)):
        charts = lida.visualize(summary=summary, 
                                goal=goals[i], 
                                    
                                library=library)
        if len(charts)>0:
            # print(charts)
            img_base64_string = charts[0].raster
            img = base64_to_image(img_base64_string)
            Visuals.append(img)
            Goals.append(goals[i])
    return Visuals,Goals
