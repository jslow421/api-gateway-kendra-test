import json
import os
import sys
import sys
import traceback
from uuid import UUID
import re

from typing import Any, Dict, Optional, List

import boto3
import langchain

import warnings

warnings.filterwarnings("ignore")

from langchain.llms.bedrock import Bedrock
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.schema import SystemMessage, HumanMessage, AIMessage, ChatMessage

# from langchain.memory import ChatMessageHistory


from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

from langchain.chat_models.bedrock import BedrockChat

from langchain.memory.buffer import ConversationBufferMemory

langchain.llm_cache = InMemoryCache()
langchain.debug = True
langchain.verbose = True

POLLY = boto3.client("polly")
TRANSLATE = boto3.client("translate")
KENDRA_CLIENT = boto3.client("kendra")
BEDROCK_CLIENT = boto3.client("bedrock-runtime")
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID", "93288901-38ed-4da9-81df-92564b243cff")

MAX_TOKENS_TO_SAMPLE: int = 2048
TEMPERATURE: float = 0.9
TOP_K: int = 250
TOP_P: float = 0.99
STOP_SEQUENCES = ["\n\nHuman:"]

MAX_HISTORY_LENGTH = 10
DEFAULT_PROMPT_TEMPLATE = """    
This conversation demonstrates the power of retrieval augmented generation.
You are the AI. You will synthesize a response from your own context and the provided documents.
Follow these rules for your responses:
    If the documents are relevant, use them to synthesize your answer.
    DO NOT MENTION THE DOCUMENTS IN YOUR ANSWER.
    For simple factual questions, respond with a single sentence.
    For more complex questions that reference specific information, you can respond with 2000s of tokens. Use complex formatting like markdown.
    If you don't know the answer to a question, truthfully reply: "I'm not sure how to answer that"
    Reply in the language the question was asked in.
Please confirm your understanding of the above requirements before I pass in the user's question and documents.

Assistant: I understand the requirements. I will synthesize responses based on the provided documents without mentioning that I am doing this. I will use simple sentences for factual questions and complex markdown formatting for more involved questions. If I don't know the answer, I will not guess. I will match the language of the question. Please provide the question and documents when ready.

Human: Here are potentially relevant documents in <documents> XML tags:
<documents>
{context}
</documents>
Answer the user's question: "{question}"

Assistant:
"""

# DEFAULT_CONDENSE_QA_TEMPLATE = """
# Human: Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}

# Assistant:
# """
# DEFAULT_CONDENSE_QA_TEMPLATE = """\
# Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
DEFAULT_CONDENSE_QA_TEMPLATE = """{chat_history}
Given the above conversation and a follow up question below, rephrase the follow up question 
to be a standalone question.

Follow Up Question: {question}"""

claude_prompt = PromptTemplate.from_template(
    """
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. If the AI does not know
the answer to a question, it truthfully says it does not know.

Current conversation:
<conversation_history>
{chat_history}
</conversation_history>

Human:
<human_reply>
{question}
</human_reply>

Assistant:
"""
)

randall_prompt = PromptTemplate.from_template(
    """
Given the above conversation and a follow up question below, rephrase the follow up question 
to be a standalone question.

Current conversation:
<conversation_history>
{chat_history}
</conversation_history>

Human:
<human_reply>
{question}
</human_reply>

Standalone Question:
Assistant:
"""
)

MAX_HISTORY_LENGTH = 10


class MessageHistory:
    def __init__(
        self,
        content: Optional[str] = None,
        role: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ):
        self.content = content
        self.role = role
        self.sources = sources


def build_chain(prompt_template, condense_qa_template):
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
    llm = Bedrock(
        client=BEDROCK_CLIENT,
        model_id="anthropic.claude-v2",
        model_kwargs={
            "max_tokens_to_sample": MAX_TOKENS_TO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "stop_sequences": ["\n\nHuman:"],
        },
    )

    retriever = AmazonKendraRetriever(
        client=KENDRA_CLIENT, index_id=KENDRA_INDEX_ID, top_k=3
    )

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)
    # standalone_question_prompt = PromptTemplate(
    #     template=condense_qa_template, input_variables=["chat_history", "question"]
    # )
    standalone_question_prompt = randall_prompt

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True,
        memory=ConversationBufferMemory(
            # Anthropic requires the prefix to be "Assistant":
            ai_prefix="Assistant"
        ),
    )
    return qa


def run_chain(chain, prompt: str, history=[]):
    print("Running chain")
    return chain({"question": prompt, "chat_history": history})


def generate_history(history):
    message_history = ChatMessageHistory()
    for item in history:
        # skip if item doesn't have the role property
        if not hasattr(item, "role") or item.role is None:
            print("Skipping item")
            continue
        # If the item is a user generated message, add it to the messages list
        if item.role == "user":
            val = item.content
            print("Found user content: " + item.content)
            message_history.add_user_message(val)
        # If the item is a system generated message, add it to the messages list
        elif item.role == "system":
            val = item.content
            print("Found system content: " + item.content)
            message_history.add_ai_message(val)
    return message_history


# def generate_history(history):
#     print("Generating history")
#     message_history = str()
#     for item in history:
#         # skip if item doesn't have the role property
#         if not hasattr(item, "role") or item.role is None:
#             continue
#         print(item.content)
#         # If the item is a user generated message, add it to the messages list
#         if item.role == "user":
#             print("Found user content: " + item.content)
#             val = item.content
#             message_history += """\n\nHuman: """ + val
#         # If the item is a system generated message, add it to the messages list
#         elif item.role == "system":
#             val = item.content
#             print("Found system content: " + item.content)
#             message_history += """\n\nAssistant: """ + val
#     return message_history


def run_chatbot_request(messages, question):
    chat_input = question
    hist = generate_history(messages)

    chat_history = hist.messages
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]

    try:
        llm_chain = build_chain(DEFAULT_PROMPT_TEMPLATE, DEFAULT_CONDENSE_QA_TEMPLATE)
    except Exception as e:
        return {"error building chain": str(e), "trace": traceback.format_exc()}
    try:
        result = run_chain(llm_chain, chat_input, chat_history)
        answer = result["answer"]
        print("Answer: " + answer)
        chat_history.append((chat_input, answer))
        document_list = []
        if "source_documents" in result:
            for d in result["source_documents"]:
                if not (d.metadata["source"] in document_list):
                    document_list.append((d.metadata["source"]))
    except Exception as e:
        return {"error running chain": str(e), "trace": traceback.format_exc()}
    # Return answer which is result, and then document list as sources
    return {"answer": answer, "sources": document_list}


def get_an_answer(messages, question):
    try:
        value = run_chatbot_request(messages, question)
        return value
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def lambda_handler(event, _):
    response = {}

    request_body = json.loads(event["body"])
    messages = request_body.get("messages", [])
    question = request_body.get("question", "")
    message_list = None
    message_list = [MessageHistory]

    for item in messages:
        mes = MessageHistory(
            item.get("content", None), item.get("role", None), item.get("sources", None)
        )
        message_list.append(mes)

    try:
        response["statusCode"] = 200
        response["headers"] = {
            "Access-Control-Allow-Origin": "*",
            "content-type": "application/json",
        }
        resp = get_an_answer(message_list, question)

        response["body"] = json.dumps(resp)

    except Exception as e:
        response["statusCode"] = 500
        response["body"] = str(e)
    return response
