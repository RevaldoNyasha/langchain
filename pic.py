from dotenv import load_dotenv

from langchain.chat_models import init_chat_model


load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

message = {
    'role':'user',
    'content':[
        {'type':'text','text':'Describe the contents of the given image.'},
        {'type':'image','url':'https://picsum.photos/200/300'}

    ]
}

response = model.invoke([message])

print(response) 