import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()

model = init_chat_model(
    model="google_genai:gemini-2.5-flash-lite",
    temperature = 0.1,
    max_tokens=400
)

conversations = [
    SystemMessage("You are cs assistant who guides revaldo nyasha a cs student who is doing cs in zimbabwe"),
    HumanMessage({"why should i focus on ai engineering to have a successfull career"}),
    AIMessage("Well this aligns with your interests aleady beacuse you know python and fast api and you already know most of the concepts "),
    HumanMessage("tell me projects i can develop or program as for my final year with this to show case my skills ")
]


# response = model.invoke(f"tell me more about {conversations}")

for chunk in model.stream("hello what is python?"):
    print(chunk.text,end='',flush=True)

# print(response)
# print("*********************************")
# print(response.content) 