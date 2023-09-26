from typing import Union

from fastapi import FastAPI
import lambda_function

from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
def read_root():
    return lambda_function.get_an_answer(
        {"body": {"chat_input": "Who was the first man on the moon?"}}
    )


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/chatbot")
# Read in the request body
def chatbot(request: dict):
    chat_input = request.get("currentMessage", "")
    response = lambda_function.get_an_answer({"body": {"chat_input": chat_input}})
    return response
