import os
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from psi import llm_automation, Linkedin_post
from openai import OpenAI
from langchain_openai import ChatOpenAI
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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
from PyPDF2 import PdfReader
import nomic
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
import shutil

import os
import sqlite3
from openai import OpenAI

## Configure OpenAI API key
openai_key = "sk-14lu6u4Zz6SWH9naxDqBM6G8hrd8QnjCjTQEll13AWT3BlbkFJ2XoxLvOf3GB5Kk0vraujlpEoimotZraw6ZNpGdNJUA"


app = FastAPI()

# CORS setup (if needed for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




os.environ['COHERE_API_KEY'] = "yTmKdlP6vaGOZ91YAlPCKqMUpvmD2rgSoZqZJRHS"
os.environ['TAVILY_API_KEY'] = "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"
nomic.cli.login("nk-TbdtpiqAFh3TRTPDLItfr6FLiUpXYb2TwapWvrEhi_g")


chat = ChatCohere(model="command-r-plus", temperature=0.3)

internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."



class TavilySearchInput(BaseModel):
    query: str = Field(description="Query to search the internet with")
internet_search.args_schema = TavilySearchInput


python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
    func=python_repl.run,
)

# from langchain_core.pydantic_v1 import BaseModel, Field
class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")
repl_tool.args_schema = ToolInput



class GathnexAIWrapper:
    def run(prompt):
        """
        This function wraps the Gathnex_AI logic, deciding whether to post on LinkedIn or
        return a normal GPT response based on the input prompt.
        """
        openai_api_key = "sk-14lu6u4Zz6SWH9naxDqBM6G8hrd8QnjCjTQEll13AWT3BlbkFJ2XoxLvOf3GB5Kk0vraujlpEoimotZraw6ZNpGdNJUA"
        linkedin_access_token = "AQXe8YJy9q9xep-UnD2k63Qaar27YGZna22eVXO8-D06SPb0A8ef2_PxFbSDvftUpfFO1AyuI1gBoN5Z-odEZo6LHEPUUx0dVVM6-hJHOyk8DXogP3mIO8UGZB-6c9QRj0xH2n1bun0TrNoP4XWHWxjfDvsANnEl5XAOknMx9iFnthR_WrzncH13OVMhxLqxrkqdKFfwQrGoU7eEhqjSftJoBJzb4Hyk0LMI2k-GAKaHWvsfehCd-_IEhAUhC_G9PPliLdggCAktWRpiqE75k04tfUf6v6IwHVC-apWJO6-G3vl6h5IAGnwxBA37pT6HT4nWYcuAhKOb34X-S07JS0anJpNC5A"

        llm = llm_automation.llm_auto(prompt, openai_api_key)
        
        if True:
            res = Linkedin_post.LinkedinAutomate(linkedin_access_token, openai_api_key,prompt).main_func()
            return llm.posted_or_not(res)
        else:
            return llm.normal_gpt()
        

# Example of how to wrap it as a tool
gathnex_tool = Tool(
    name="gathnex_ai_tool",
    description="you can post in linkedin  ",
    func=GathnexAIWrapper.run)


class Hello(BaseModel):
    code: str = Field(description="you can post in linkedin")
gathnex_tool.args_schema = Hello



# Create the prompt
prompt = ChatPromptTemplate.from_template("{input}")

class gmail:
    def mailing(body):
        print(body)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587  # Gmail's SMTP port for TLS

        # Login credentials
        your_email = "jaha22049.cs@rmkec.ac.in"
        your_password ="arpu mcrc bxrn idit"

        # Create the message
        message = MIMEMultipart()
        message["From"] = your_email
        message["To"] = "iamrengoku04@gmail.com"
        message["Subject"] = "mail"
        body = body

        # Attach the body to the email

        message.attach(MIMEText(body, "plain"))

        # Send the email
        try:
            # Start the server and log in
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # Enable security (TLS)
            server.login(your_email, your_password)
            
            # Send the email
            server.sendmail(your_email,message.as_string())
            
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")
        finally:
            server.quit()


class vankkam(BaseModel):
    code: str = Field(description="you can mail a person")
    to: str = Field(description="The recipient's email address.")

mail_tool=Tool(
    name="mailer",
    description="you can mail a person",
    func=gmail.mailing,
    args_schema = vankkam
)



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama3.1")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain 

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text +=page.extract_text()
    return text

docs=get_pdf_text("E:\Linkedin_Automation_with_Generative_AI\Trumio (1).pdf")

text_chunks = get_text_chunks(docs)
                

vectorstore = get_vectorstore(text_chunks)


def one(prompt):
        conversation_chain = get_conversation_chain(vectorstore)
        response = conversation_chain(prompt)
        return response 

                
class rag:
    def rag_runner(prompt):
        response=one(prompt)
        return response
    
class ragger(BaseModel):
    code: str = Field(description="you are documentation agent which is supposed to read the document and summerize")

rag_tool=Tool(
    name="rag_tool",
    description="you are documentation agent which is supposed to read the document and summerize",
    func=rag.rag_runner,
    arg_schema=ragger
)


def get_openai_response(question):
    model = "gpt-4o-mini"
    client = OpenAI(api_key=openai_key)
    DEFAULT_SYSTEM_PROMPT = """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
    SECTION \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ;
    \nExample 2 - Tell me all the students studying in Data Science class?, 
    the SQL command will be something like this SELECT * FROM STUDENT 
    where CLASS="Data Science"; 
    also the sql code should not have ``` in beginning or end and sql word in output
    """
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )
    return response.choices[0].message.content

def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows


class tracker:
    def tracking(prompt):
        responsive=get_openai_response(prompt)
        sql_result = read_sql_query(responsive, "student.db")
        return sql_result
    
class track(BaseModel):
    code: str = Field(description="you can access the database and answer for the quires like student tracking")

track_tool=Tool(
    name="tracker_tool",
    description="you can access the database and answer for the quires like student tracking",
    func=tracker.tracking,
    arg_schema=track
)


# Create the ReAct agent
agent = create_cohere_react_agent(
    llm=chat,
    tools=[internet_search, repl_tool,gathnex_tool,mail_tool,rag_tool,track_tool],
    prompt=prompt,
)


agent_executor = AgentExecutor(agent=agent, tools=[internet_search, repl_tool, gathnex_tool,mail_tool,rag_tool,track_tool], verbose=True)


class ChatbotRequest(BaseModel):
    input_text: str

# Chatbot endpoint
@app.post("/ask")
async def chatbot(input_text: str = Form(...), file: UploadFile = File(None)):
    try:
        # Save the file if it's uploaded
        if file:
            file_location = "Trumio (1).pdf"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"File saved as {file_location}")

        # Handle the chatbot input_text (logic remains unchanged)
        result = agent_executor.invoke({"input": input_text})
        return {"response": result}
    except Exception as e:
        print(f"Error details: {e}")  # Print the error details for debugging
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Run FastAPI server (if running locally)
if __name__ == "__mains__":
    uvicorn.run(app, host="0.0.0.0", port=8000)