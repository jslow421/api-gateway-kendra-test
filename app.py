from typing import Literal
import uuid
import os

import boto3
import langchain
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()
langchain.debug = True
langchain.verbose = True

POLLY = boto3.client("polly")
TRANSLATE = boto3.client("translate")
KENDRA_CLIENT = boto3.client("kendra")
BEDROCK_CLIENT = boto3.client("bedrock")
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID", "dc60affd-f6aa-4ae2-9b4b-a0dc7da5bbb6")
MAX_TOKENS_TO_SAMPLE: int = 2048
TEMPERATURE: float = 0.9
TOP_K: int = 250
TOP_P: float = 0.99
STOP_SEQUENCES = ["\n\nHuman:"]

LANGUAGES_DATA = [
    {"LanguageName": "Spanish", "LanguageCode": "es"},
    {"LanguageName": "Afrikaans", "LanguageCode": "af"},
    {"LanguageName": "Albanian", "LanguageCode": "sq"},
    {"LanguageName": "Amharic", "LanguageCode": "am"},
    {"LanguageName": "Arabic", "LanguageCode": "ar"},
    {"LanguageName": "Armenian", "LanguageCode": "hy"},
    {"LanguageName": "Auto", "LanguageCode": "auto"},
    {"LanguageName": "Azerbaijani", "LanguageCode": "az"},
    {"LanguageName": "Bengali", "LanguageCode": "bn"},
    {"LanguageName": "Bosnian", "LanguageCode": "bs"},
    {"LanguageName": "Bulgarian", "LanguageCode": "bg"},
    {"LanguageName": "Canadian French", "LanguageCode": "fr-CA"},
    {"LanguageName": "Catalan", "LanguageCode": "ca"},
    {"LanguageName": "Chinese", "LanguageCode": "zh"},
    {"LanguageName": "Chinese Traditional", "LanguageCode": "zh-TW"},
    {"LanguageName": "Croatian", "LanguageCode": "hr"},
    {"LanguageName": "Czech", "LanguageCode": "cs"},
    {"LanguageName": "Danish", "LanguageCode": "da"},
    {"LanguageName": "Dari", "LanguageCode": "fa-AF"},
    {"LanguageName": "Dutch", "LanguageCode": "nl"},
    {"LanguageName": "English", "LanguageCode": "en"},
    {"LanguageName": "Estonian", "LanguageCode": "et"},
    {"LanguageName": "Finnish", "LanguageCode": "fi"},
    {"LanguageName": "French", "LanguageCode": "fr"},
    {"LanguageName": "Georgian", "LanguageCode": "ka"},
    {"LanguageName": "German", "LanguageCode": "de"},
    {"LanguageName": "Greek", "LanguageCode": "el"},
    {"LanguageName": "Gujarati", "LanguageCode": "gu"},
    {"LanguageName": "Haitian Creole", "LanguageCode": "ht"},
    {"LanguageName": "Hausa", "LanguageCode": "ha"},
    {"LanguageName": "Hebrew", "LanguageCode": "he"},
    {"LanguageName": "Hindi", "LanguageCode": "hi"},
    {"LanguageName": "Hungarian", "LanguageCode": "hu"},
    {"LanguageName": "Icelandic", "LanguageCode": "is"},
    {"LanguageName": "Indonesian", "LanguageCode": "id"},
    {"LanguageName": "Irish", "LanguageCode": "ga"},
    {"LanguageName": "Italian", "LanguageCode": "it"},
    {"LanguageName": "Japanese", "LanguageCode": "ja"},
    {"LanguageName": "Kannada", "LanguageCode": "kn"},
    {"LanguageName": "Kazakh", "LanguageCode": "kk"},
    {"LanguageName": "Korean", "LanguageCode": "ko"},
    {"LanguageName": "Latvian", "LanguageCode": "lv"},
    {"LanguageName": "Lithuanian", "LanguageCode": "lt"},
    {"LanguageName": "Macedonian", "LanguageCode": "mk"},
    {"LanguageName": "Malay", "LanguageCode": "ms"},
    {"LanguageName": "Malayalam", "LanguageCode": "ml"},
    {"LanguageName": "Maltese", "LanguageCode": "mt"},
    {"LanguageName": "Marathi", "LanguageCode": "mr"},
    {"LanguageName": "Mexican Spanish", "LanguageCode": "es-MX"},
    {"LanguageName": "Mongolian", "LanguageCode": "mn"},
    {"LanguageName": "Norwegian", "LanguageCode": "no"},
    {"LanguageName": "Pashto", "LanguageCode": "ps"},
    {"LanguageName": "Persian", "LanguageCode": "fa"},
    {"LanguageName": "Polish", "LanguageCode": "pl"},
    {"LanguageName": "Portugal Portuguese", "LanguageCode": "pt-PT"},
    {"LanguageName": "Portuguese", "LanguageCode": "pt"},
    {"LanguageName": "Punjabi", "LanguageCode": "pa"},
    {"LanguageName": "Romanian", "LanguageCode": "ro"},
    {"LanguageName": "Russian", "LanguageCode": "ru"},
    {"LanguageName": "Serbian", "LanguageCode": "sr"},
    {"LanguageName": "Sinhala", "LanguageCode": "si"},
    {"LanguageName": "Slovak", "LanguageCode": "sk"},
    {"LanguageName": "Slovenian", "LanguageCode": "sl"},
    {"LanguageName": "Somali", "LanguageCode": "so"},
    {"LanguageName": "Swahili", "LanguageCode": "sw"},
    {"LanguageName": "Swedish", "LanguageCode": "sv"},
    {"LanguageName": "Tagalog", "LanguageCode": "tl"},
    {"LanguageName": "Tamil", "LanguageCode": "ta"},
    {"LanguageName": "Telugu", "LanguageCode": "te"},
    {"LanguageName": "Thai", "LanguageCode": "th"},
    {"LanguageName": "Turkish", "LanguageCode": "tr"},
    {"LanguageName": "Ukrainian", "LanguageCode": "uk"},
    {"LanguageName": "Urdu", "LanguageCode": "ur"},
    {"LanguageName": "Uzbek", "LanguageCode": "uz"},
    {"LanguageName": "Vietnamese", "LanguageCode": "vi"},
    {"LanguageName": "Welsh", "LanguageCode": "cy"},
]


def synthesize_speech(text):
    response = POLLY.synthesize_speech(
        Engine="neural",
        Text=text,
        OutputFormat="mp3",
        VoiceId="Matthew",
    )
    return response


# Consider adding this to prompt for some use cases -
#
# When you reply, first find content in the documents relevant to the user's question and write them down word for word inside <thinking></thinking> XML tags.
# This is a space for you to write down relevant content and will not be shown to the user.
# Once you are done extracting relevant quotes, answer the question.
# Put your answer to the user inside <answer></answer> XML tags.


USER_ICO = "user-icon.png"
AI_ICON = "ai-icon.png"
MAX_HISTORY_LENGTH = 10
DEFAULT_PROMPT_TEMPLATE = """\
Human: This conversation demonstrates the power of retrieval augmented generation.
You are the AI. You will synthesize a response from your own context and the provided documents.
Follow these rules for your responses:
* If the documents are relevant, use them to synthesize your answer.
* DO NOT MENTION THE DOCUMENTS IN YOUR ANSWER.
* For simple factual questions, respond with a single sentence.
* For more complex questions that reference specific information, you can respond with 1000s of tokens. Use complex 
formatting like markdown.
* If you don't know the answer to a question, truthfully reply: "I'm not sure how to answer that, but I'd be happy to 
introduce you to a Caylent engineer who can help."
* Reply in the language the question was asked in.

Please confirm your understanding of the above requirements before I pass in the user's question and documents.

Assistant: I understand the requirements. I will synthesize responses based on the provided documents without mentioning 
that I am doing this. I will use simple sentences for factual questions and complex markdown formatting for more 
involved questions. If I don't know the answer, I will direct you to a Caylent engineer. I will match the language of 
the question. Please provide the question and documents when ready.

Human: Here are potentially relevant documents in <documents> XML tags:
<documents>
{context}
</documents>
Answer the user's question: "{question}"

Assistant:"""

DEFAULT_CONDENSE_QA_TEMPLATE = """\
Human: Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

# st.session_state.setdefault("user_id", str(uuid.uuid4()))
# st.session_state.setdefault("chat_history", [])
# st.session_state.setdefault("chats", [{"id": 0, "question": "", "answer": ""}])
# st.session_state.setdefault("questions", [])
# st.session_state.setdefault("answers", [])
# st.session_state.setdefault("chat_input", "")
# st.session_state.setdefault("prompt_template", DEFAULT_PROMPT_TEMPLATE)
# st.session_state.setdefault("condense_qa_template", DEFAULT_CONDENSE_QA_TEMPLATE)
# st.session_state.setdefault("translate_language", "es")


def build_chain(prompt_template, condense_qa_template):
    llm = Bedrock(
        client=BEDROCK_CLIENT,
        model_id="anthropic.claude-v2",
        model_kwargs={
            "max_tokens_to_sample": MAX_TOKENS_TO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "stop_sequences": STOP_SEQUENCES,
        },
    )

    retriever = AmazonKendraRetriever(
        client=KENDRA_CLIENT, index_id=KENDRA_INDEX_ID, top_k=3
    )

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return qa


def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})
