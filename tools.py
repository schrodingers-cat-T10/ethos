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

os.environ['COHERE_API_KEY'] = "yTmKdlP6vaGOZ91YAlPCKqMUpvmD2rgSoZqZJRHS"
os.environ['TAVILY_API_KEY'] = "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"


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
repl_tool.name = "python_interpreter"

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
        openai_api_key="openai_key",
        linkedin_access_token="limkedinidkey"

        llm = llm_automation.llm_auto(prompt, openai_api_key)
        
        if llm.intent_indentifier() == "#Post":
            url = llm.prompt_link_capturer()
            res = Linkedin_post.LinkedinAutomate(linkedin_access_token, url, openai_api_key).main_func()
            return llm.posted_or_not(res)
        else:
            return llm.normal_gpt()
        


# Example of how to wrap it as a tool
gathnex_tool = Tool(
    name="gathnex_ai_tool",
    description="you have access to my linkedin post on it ",
    func=GathnexAIWrapper.run)
gathnex_tool.name = "gathnex_ai_tool"

class Hello(BaseModel):
    code: str = Field(description="you have access to my linkedin  post on it")
gathnex_tool.args_schema = Hello

# Create the prompt
prompt = ChatPromptTemplate.from_template("{input}")


# Create the ReAct agent
agent = create_cohere_react_agent(
    llm=chat,
    tools=[internet_search, repl_tool,gathnex_tool],
    prompt=prompt,
)


agent_executor = AgentExecutor(agent=agent, tools=[internet_search, repl_tool, gathnex_tool], verbose=True)



hello=agent_executor.invoke({
    "input": "make a linkedin post for gen ai and post it in linkedin",
})
