from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

#pip install langchain-google-genai 


@dataclass
class Context:
    user_id:str


@dataclass
class ResponseFormat:
    summary:str
    temperature_celsius:float
    temperature_fahrenheit:float 
    humidity:float

@tool('locate_user',description="Look up a users city based on the context")
def locate_user(runtime:ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'H':
            return 'Harare'
        case 'B':
            return 'Bulawayo'
        case 'C':
            return 'Chinhoyi'
        case _:
            return 'Unknown'

@tool(
    "get_weather",
    description="Return weather information for a given city",
    return_direct=False
)
def get_weather(city: str):

    response = requests.get(
        f"https://wttr.in/{city}?format=j1"
    )

    response.raise_for_status()

    return response.json()

checkpointer = InMemorySaver()

#agent for this 
agent = create_agent(
    model="google_genai:gemini-2.5-flash-lite",
    tools=[get_weather,locate_user],
    system_prompt="""
    You are a helpful weather assistant.

    When answering weather questions:
    1. Identify the user's city using the locate_user tool.
    2. Get the current weather using the get_weather tool.
    3. Give a detailed but easy-to-understand summary.
    4. The summary should mention:
    - Whether it is sunny, cloudy, rainy, etc.
    - The temperature
    - How the weather feels
    - Humidity
    - Any useful advice for the user

    Explain things in a hilarious and fun way so a student can understand.

    Do not make the summary just one or two words.
    """,
    context_schema = Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
) 


config = {'configurable':{'thread_id':'1'}}

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like today'}
    ]},
    config = config,
    context =  Context(user_id='H')
)

# print(response)
print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)



