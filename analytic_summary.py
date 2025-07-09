import os
import asyncio
from typing import List, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class State(TypedDict):
    contents: List[str]
    index: int
    summary: str

class SummaryGenerator:
    def __init__(self):
        self.llm = init_chat_model("llama3-8b-8192", model_provider="groq")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=200,
        )
        
        self.summarize_prompt = ChatPromptTemplate.from_messages([
            ("human", "Write a concise analytical report of the following: {context}")
        ])
        
        self.refine_template = """
        Produce a final analytical report by capturing insights of the data. 
        The final report should contain the following sections:
        1. Descriptive analysis 
        2. Diagnostic analysis
        3. Outlier analysis
        4. What are the insights?
        5. Key takeaways
        6. Future recommendations to clients
        
        Existing summary up to this point:
        {existing_answer}
        
        New context:
        ------------
        {context}
        ------------
        
        Given the new context, refine the original summary.
        """
        
        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("human", self.refine_template)
        ])

    def load_data(self, file_path='data_results.txt'):
        """Load and split the document content"""
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            self.documents = self.splitter.split_documents(docs)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def initialize_chains(self):
        """Initialize the processing chains"""
        self.initial_summary_chain = (self.summarize_prompt 
                                    | self.llm 
                                    | StrOutputParser())
        self.refine_summary_chain = (self.refine_prompt 
                                   | self.llm 
                                   | StrOutputParser())

    async def generate_report(self) -> str:
        """Main method to generate the final report"""
        if not hasattr(self, 'documents') or not self.documents:
            return "Error: No documents loaded"
            
        # Build the processing graph
        graph = StateGraph(State)
        
        # Define async node functions
        async def generate_initial_summary(state: State, config: RunnableConfig):
            summary = await self.initial_summary_chain.ainvoke(
                {"context": state["contents"][0]},
                config
            )
            return {"summary": summary, "index": 1}

        async def refine_summary(state: State, config: RunnableConfig):
            summary = await self.refine_summary_chain.ainvoke(
                {
                    "existing_answer": state["summary"],
                    "context": state["contents"][state["index"]]
                },
                config
            )
            return {"summary": summary, "index": state["index"] + 1}

        def should_refine(state: State) -> Literal["refine_summary", END]:
            return END if state["index"] >= len(state["contents"]) else "refine_summary"

        # Build the graph
        graph.add_node("generate_initial_summary", generate_initial_summary)
        graph.add_node("refine_summary", refine_summary)
        graph.add_edge(START, "generate_initial_summary")
        graph.add_conditional_edges("generate_initial_summary", should_refine)
        graph.add_conditional_edges("refine_summary", should_refine)
        
        compiled_app = graph.compile()
        
        # Prepare document chunks (every 30th document)
        contents = [
            doc.page_content 
            for i, doc in enumerate(self.documents) 
            if i % 30 == 0
        ]
        
        # Run the graph
        final_result = await compiled_app.ainvoke(
            {"contents": contents},
            config=RunnableConfig(recursion_limit=35)
        )
        
        return final_result.get("summary", "No summary generated")

# Usage example
async def main():
    generator = SummaryGenerator()
    
    if not generator.load_data("data_results.txt"):
        return
        
    generator.initialize_chains()
    
    try:
        summary = await generator.generate_report()
        print("Final Summary:")
        print(summary)
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    asyncio.run(main())