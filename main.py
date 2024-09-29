import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_nomic import NomicEmbeddings
import nomic
from langchain.agents import Tool, AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI()

# Allow CORS for your React app (adjust this to your React app's URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to your React app URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key for Nomic
nomic.cli.login(os.getenv('NOMIC_API_KEY'))

# Load Cohere API key
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")

# Tavily Search API key
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Function to create vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create conversation chain
def get_conversation_chain(vectorstore, model):
    llm = Ollama(model=model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Define request model for the chatbot
class QuestionRequest(BaseModel):
    question: str
    model: str

# Define request model for Python code execution
class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")

# Chatbot agent setup
chat = ChatCohere(model="command-r-plus", temperature=0.3)

# Tavily search tool
internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns relevant document snippets for a textual query retrieved from the internet."
internet_search.args_schema = BaseModel

# Python REPL tool
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_interpreter",
    description="Executes Python code in a static sandbox and returns the result.",
    func=python_repl.run,
)
repl_tool.args_schema = ToolInput

# ReAct agent setup
prompt = ChatPromptTemplate.from_template("{input}")
agent = create_cohere_react_agent(
    llm=chat,
    tools=[internet_search, repl_tool],
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=[internet_search, repl_tool], verbose=True)

# FastAPI route to handle questions for the chatbot agent
@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    # Load text data (if any specific document is being used)
    with open("truimo.txt", "r") as file:
        data = file.read()

    # Process text data and create vectorstore
    text_chunks = get_text_chunks(data)
    vectorstore = get_vectorstore(text_chunks)

    # Create the conversation chain using the model provided
    conversation_chain = get_conversation_chain(vectorstore, request.model)

    # Get the response from the conversation chain
    response = conversation_chain(request.question)
    return {"answer": response['answer']}

# FastAPI route to handle queries for the chatbot with Cohere agent and tools
@app.post("/execute-agent/")
async def execute_agent(query: str):
    # Run the agent with the input query
    result = agent_executor.invoke({
        "input": query,
    })
    return {"result": result}

# Start the FastAPI app
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
