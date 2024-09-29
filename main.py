import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_nomic import NomicEmbeddings
from langchain.agents import Tool, AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI()

# Allow CORS for your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Nomic API key
nomic_api_key = os.getenv('NOMIC_API_KEY')
# Load Cohere API key
cohere_api_key = os.getenv("COHERE_API_KEY")
# Load Tavily API key
tavily_api_key = os.getenv("TAVILY_API_KEY")
# Load LinkedIn access token
linkedin_access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
# Load Gmail API key
gmail_api_key = os.getenv('GMAIL_API_KEY')

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

# Fetch LinkedIn profile details
class LinkedInProfileTool:
    def __init__(self):
        self.name = "fetch_linkedin_profile"
        self.description = "Fetches LinkedIn profile details based on user input (name or company)."
    
    def run(self, input: str):
        headers = {"Authorization": f"Bearer {linkedin_access_token}"}
        url = f"https://api.linkedin.com/v2/people/(name:{input})"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json()}

linkedin_tool = LinkedInProfileTool()

# Update a LinkedIn post
class LinkedInPostUpdateTool:
    def __init__(self):
        self.name = "update_linkedin_post"
        self.description = "Updates a LinkedIn post about the user context using API access."
    
    def run(self, post_id: str, content: str):
        headers = {
            "Authorization": f"Bearer {linkedin_access_token}", 
            "Content-Type": "application/json"
        }
        url = f"https://api.linkedin.com/v2/posts/{post_id}"
        data = {"content": content}
        response = requests.put(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return {"success": "Post updated successfully."}
        else:
            return {"error": response.json()}

linkedin_post_update_tool = LinkedInPostUpdateTool()

# Web scraping tool
class WebScrapingTool:
    def __init__(self):
        self.name = "web_scrape"
        self.description = "Scrapes websites for specific text or patterns and returns results."
    
    def run(self, url: str, pattern: str):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        results = soup.find_all(string=pattern)
        return {"results": [result for result in results]}

web_scraping_tool = WebScrapingTool()

# Gmail content generator tool
class GmailContentGeneratorTool:
    def __init__(self):
        self.name = "generate_gmail_content"
        self.description = "Generates and sends Gmail content to a provided email address."
    
    def run(self, email: str, subject: str, body: str):
        service = build('gmail', 'v1', developerKey=gmail_api_key)
        message = {
            'raw': f'From: your_email@gmail.com\nTo: {email}\nSubject: {subject}\n\n{body}'
        }
        service.users().messages().send(userId='me', body=message).execute()
        return f"Email sent to {email}"

gmail_content_generator_tool = GmailContentGeneratorTool()

# Internet data analyzer tool
class InternetDataAnalyzerTool:
    def __init__(self):
        self.name = "internet_data_analyzer"
        self.description = "Scrapes the internet, gathers data, and analyzes with command R+."
    
    def run(self, query: str):
        # Implement scraping and analyzing logic here
        return {"result": f"Scraped data for query: {query} and analyzed it."}

internet_data_analyzer_tool = InternetDataAnalyzerTool()

# Meeting minutes generator tool
class MeetingMinutesGeneratorTool:
    def __init__(self):
        self.name = "generate_meeting_minutes"
        self.description = "Generates meeting minutes from user-provided video/audio."
    
    def run(self, audio_path: str):
        # Placeholder for audio processing logic
        return {"result": f"Generated meeting minutes from audio: {audio_path}"}

meeting_minutes_generator_tool = MeetingMinutesGeneratorTool()

# CSV analyzer tool
class CSVAnalyzerTool:
    def __init__(self):
        self.name = "analyze_csv"
        self.description = "Analyzes CSV files and provides trained ML/DL models based on user needs."
    
    def run(self, csv_path: str):
        df = pd.read_csv(csv_path)
        summary = df.describe()  # Returns statistical summary
        return summary.to_json()  # Return as JSON for easier consumption

csv_analyzer_tool = CSVAnalyzerTool()

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

# ReAct agent setup
prompt = ChatPromptTemplate.from_template("{input}")
agent = create_cohere_react_agent(
    llm=chat,
    tools=[
        internet_search,
        repl_tool,
        linkedin_tool,
        linkedin_post_update_tool,
        web_scraping_tool,
        gmail_content_generator_tool,
        internet_data_analyzer_tool,
        meeting_minutes_generator_tool,
        csv_analyzer_tool
    ],
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
