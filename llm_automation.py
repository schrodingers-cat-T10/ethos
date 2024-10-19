import re
from openai import OpenAI

class llm_auto:

    def __init__(self, prompt, openai_api):
        print(prompt)
        self.prompt = prompt
        self.openai_api = openai_api

    def intent_identifier(self):
        model = "gpt-4o-mini"
        client = OpenAI(api_key=self.openai_api)
        DEFAULT_SYSTEM_PROMPT = '''Classify the assigned task as #Post or other intents based on the prompt.'''
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "Classify the prompt as '#Post': " + self.prompt},
            ]
        )
        return response.choices[0].message.content

    def normal_gpt(self):
        model = "gpt-4o-mini"
        client = OpenAI(api_key=self.openai_api)
        DEFAULT_SYSTEM_PROMPT = '''
        You are Gathnex, an intelligent assistant dedicated to providing effective solutions. 
        Your responses will include emojis to add a friendly and engaging touch. 😊 
        Analyze user queries and provide clear and practical answers, incorporating emojis to enhance the user experience. 
        Focus on delivering solutions that are accurate, actionable, and helpful. If additional information is required for a more precise solution, 
        politely ask clarifying questions. Your goal is to assist users by providing effective and reliable solutions to their queries. 🌟'''
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": self.prompt},
            ]
        )
        return response.choices[0].message.content

    def posted_or_not(self, response_status):
        client = OpenAI(api_key=self.openai_api)
        model = "gpt-4o-mini"
        DEFAULT_SYSTEM_PROMPT = "You are an assistant tasked with informing the user about LinkedIn post status."

        if response_status == "<Response [201]>":
            friendly_message = '''Tell the user the LinkedIn post was successfully posted with friendly emojis.'''
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": friendly_message},
                ]
            )
            return response.choices[0].message.content
        else:
            error_message = '''Tell the user the LinkedIn post was not successfully posted. Advise them to check the access tokens and hyperparameters with a sad emoji.'''
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": error_message},
                ]
            )
            return response.choices[0].message.content
