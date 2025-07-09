# 🧠 7th Sense of Data-Driven Decision Mastery

An advanced AI-powered Streamlit application that enables business users and analysts to analyze structured data through natural language queries, auto-generated SQL, visualizations, and human-style summaries using LLM agents structured via LangGraph.

---

## 🚀 Features

- 📂 File Upload: Upload `.csv` or `.xlsx` data files  
- 🔍 EDA Chatbot: Interact with your dataset using PandasAI (Gemini/Groq/LLMs)  
- 🧠 LLM-Powered Analytics Report: Generates human-style summaries of statistical insights  
- 🤖 Graph-Based Q&A Agent: Converts questions into SQL, generates visualizations, and returns interpretable results using LangGraph  
- 📸 Quick Insights: Auto-generated business goals with relevant visualizations via LIDA  
- 📊 Custom Visualization Code Generator: Builds dynamic Plotly code from user intent  
- 🗃️ SQLite Integration: Automatically transforms uploaded CSV into SQLite for querying  

---

## 🧠 Graph-Based Agent Design (LangGraph)

This project features a LangGraph-based LLM pipeline that routes user prompts through a state-aware graph composed of reasoning nodes. It decomposes user questions into SQL and visualization tasks and follows a conditional logic path for execution and self-repair.

### ⚙️ Agent Workflow

```
START
↓
[ preprocess_input ] → Extracts SQL & visualization tasks from user input
↓
[ generate_query ] → LLM generates dialect-specific SQL query
↓
[ execute_query ] → Executes SQL via SQLite and returns result
↓
[ generate_answer ] → LLM explains the query result in human-readable text
↓
(If vis_task exists)
[ generate_code ] → LLM generates Plotly Python code using the result
↓
[ execute_code ] → Runs the code and renders the chart
↓
END
```

---

### ✅ Features of This Graph-Based Agent

- Modular execution path with fallback loops  
- Uses structured types (Pydantic) to pass query/code/answer  
- Handles SQL errors and Python errors with recovery  
- Visualization generation is optional and only triggered if `vis_task` exists  
- Supports multiple LLMs (`openai`, `groq`, `llama`) and dynamically chooses between them  

---

## 🏁 Getting Started

### ✅ Prerequisites

- Python 3.9+  
- pip or Conda  
- OpenAI and Groq API keys  

---

### 📦 Installation

```bash
git clone https://github.com/Abubacker10/7th-sense-data-driven-decision-mastery.git
cd 7th-sense-data-driven-decision-mastery
pip install -r requirements.txt
```

---

### 🔐 .env File Setup

```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

Make sure `.env` is included in `.gitignore`:

```gitignore
.env
```

---

## ▶️ Running the App

```bash
streamlit run interface.py
```

Then go to [http://localhost:8501](http://localhost:8501)

---

## 🧭 App Pages Overview

| Page                | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| 📁 File Upload      | Upload dataset and auto-trigger analysis pipeline                      |
| 📈 Analytics Report | Generates descriptive summary of uploaded data using RAG/LLM           |
| 📊 EDA Chatbot      | Ask questions via Gemini/Groq using PandasAI                           |
| 💬 Q/A Agent        | LangGraph-powered agent interprets, answers, and optionally visualizes |
| 📸 Quick Insights   | Business goal–oriented chart panels using LIDA                         |
| 🤖 Chatbot          | Structured SQL+viz reasoning using LangGraph + ChatLLM                 |
| 🔓 Logout           | Ends the session                                                       |

---

## 💡 Example Use Cases

- “Which region had the highest sales last quarter?”
- “Visualize monthly revenue trends using a bar chart”
- “Which product had the lowest return rate?”
- “Show top 5 customer segments by profit and chart it”
- “How did sales change over time in 2023?”

---

## 📂 Project Structure

```
📦 root/
 ┣ interface.py             → Main Streamlit app
 ┣ agents.py                → LangGraph agent (SQL + Viz + Answer)
 ┣ lida_agent.py            → Visual storytelling for business goals
 ┣ QA_agent.py              → Gemini/PandasAI-based data Q&A
 ┣ Data_Analysis.ipynb      → Notebook for EDA
 ┣ lightning_rag.py         → RAG pipeline for report generation
 ┣ data.csv                 → Uploaded dataset
 ┣ data_results.txt         → Text results used by summary agent
 ┣ temp.db                  → SQLite database
 ┣ requirements.txt         → Python dependencies
 ┣ .env                     → Your API keys (excluded from Git)
 ┗ .gitignore               → Prevents secrets and temp files from pushing
```

---

## 🔐 Security

- Use `.env` to store secrets
- Add `.env` to `.gitignore`
- Rotate keys if accidentally pushed
- GitHub Push Protection will block you from committing secrets

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 📄 License

MIT License © 2025 Abubacker S

---

## 🙏 Acknowledgements

- [LangChain / LangGraph](https://www.langchain.com/langgraph)
- [Streamlit](https://streamlit.io)
- [PandasAI](https://github.com/gventuri/pandas-ai)
- [OpenAI](https://platform.openai.com)
- [Groq](https://console.groq.com)
- [Plotly](https://plotly.com/python/)
- [LIDA by Salesforce](https://github.com/salesforce/LIDA)