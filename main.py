import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

#pip install langchain-google-genai 

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

agent = create_agent(
    model="google_genai:gemini-2.5-flash-lite",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
) 

response = agent.invoke({
    "messages":[
        {'role':'user', 'content':'What is the weather like today in harare'}
    ]
})

# print(response)
print(response['messages'][-1].content)