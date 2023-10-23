from calendar import c
from typing import Union, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import langchain_handler

from fastapi.middleware.cors import CORSMiddleware

from python.langchain.schema import chat_history

app = FastAPI()
origins = [
    "http://localhost:3000",  # Update with the actual origin of your JavaScript frontend
    "https://yourdomain.com",  # Add additional allowed origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict this to specific HTTP methods
    allow_headers=["*"],  # You can restrict this to specific headers
)


class MessageHistory(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    sources: Optional[List[str]] = None


class ChatInput(BaseModel):
    question: str
    messages: List[MessageHistory]


@app.get("/")
def read_root():
    return langchain_handler.get_an_answer(
        {"body": {"chat_input": "Who was the first man on the moon?"}}
    )


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/chatbot")
# Read in the request body
def chatbot(request_body: ChatInput):
    chat_input = request_body.question
    chat_history = request_body.messages
    response = langchain_handler.get_an_answer(
        {"body": {"chat_input": chat_input, "chat_history": chat_history}}
    )
    print("Completed....")
    print("response: ", response)
    return response


@app.post("/stream_text")
async def stream_text():
    return StreamingResponse(langchain_handler.generate_text(), media_type="text/plain")


@app.post("/chatbotstream")
# Read in the request body
async def chatbot(request_body: ChatInput):
    chat_input = request_body.question
    # response = await lambda_function.get_an_answer({"body": {"chat_input": chat_input}})
    return StreamingResponse(
        langchain_handler.get_an_answer({"body": {"chat_input": chat_input}}),
        media_type="text/plain",
    )
