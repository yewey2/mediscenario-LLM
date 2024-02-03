from openai import OpenAI
import streamlit as st
import streamlit.components.v1 as components
import datetime


## Firestore ??
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


## ----------------------------------------------------------------
## LLM Part
import openai
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
import tiktoken
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import langchain_community.embeddings.huggingface
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory #, ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory

import os, dotenv
from dotenv import load_dotenv
load_dotenv()

if not os.path.isdir("./.streamlit"):
    os.mkdir("./.streamlit")
    print('made streamlit folder')
if not os.path.isfile("./.streamlit/secrets.toml"):
    with open("./.streamlit/secrets.toml", "w") as f:
        f.write(os.environ.get("STREAMLIT_SECRETS"))
    print('made new file')
    

import db_firestore as db

## Load from streamlit!!
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or st.secrets["HF_TOKEN"]
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["FIREBASE_CREDENTIAL"] = os.environ.get("FIREBASE_CREDENTIAL") or st.secrets["FIREBASE_CREDENTIAL"]



st.title("UAT for PatientLLM and GraderLLM")

## Hardcode indexes for now, 
indexes = """Bleeding
ChestPain
Dysphagia
Headache
ShortnessOfBreath
Vomiting
Weakness
Weakness2""".split("\n")

if "selected_index" not in st.session_state:
    st.session_state.selected_index = 3
    
if "index_selectbox" not in st.session_state:
    st.session_state.index_selectbox = "Headache"

index_selectbox = st.selectbox("Select index",indexes, index=int(st.session_state.selected_index))

if index_selectbox != indexes[st.session_state.selected_index]:
    st.session_state.selected_index = indexes.index(index_selectbox)
    st.session_state.index_selectbox = index_selectbox
    del st.session_state["store"]
    del st.session_state["store2"]
    del st.session_state["retriever"]
    del st.session_state["retriever2"]
    del st.session_state["chain"]
    del st.session_state["chain2"]



if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

if "messages_1" not in st.session_state:
    st.session_state.messages_1 = []

if "messages_2" not in st.session_state:
    st.session_state.messages_2 = []

# if "start_time" not in st.session_state:
#     st.session_state.start_time = None

if "active_chat" not in st.session_state:
    st.session_state.active_chat = 1

model_name = "bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
# model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceBgeEmbeddings(
        # model_name=model_name, 
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs)
embeddings = st.session_state.embeddings
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
llm = st.session_state.llm
if "llm_i" not in st.session_state:
    st.session_state.llm_i = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
llm_i = st.session_state.llm_i
if "llm_gpt4" not in st.session_state:
    st.session_state.llm_gpt4 = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
llm_gpt4 = st.session_state.llm_gpt4

## ------------------------------------------------------------------------------------------------
## Patient part

index_name = f"indexes/{st.session_state.index_selectbox}/QA"

if "store" not in st.session_state:
    st.session_state.store = db.get_store(index_name, embeddings=embeddings)
store = st.session_state.store

if "TEMPLATE" not in st.session_state:
    with open('templates/patient.txt', 'r') as file: 
        TEMPLATE = file.read()
    st.session_state.TEMPLATE = TEMPLATE

with st.expander("Patient Prompt"):
    TEMPLATE = st.text_area("Patient Prompt", value=st.session_state.TEMPLATE)

prompt = PromptTemplate(
    input_variables = ["question", "context"],
    template = TEMPLATE
)
if "retriever" not in st.session_state:
    st.session_state.retriever = store.as_retriever(search_type="similarity", search_kwargs={"k":2})
retriever = st.session_state.retriever

def format_docs(docs):
    return "\n--------------------\n".join(doc.page_content for doc in docs)


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        llm=llm, memory_key="chat_history", input_key="question", 
        k=5, human_prefix="student", ai_prefix="patient",)
memory = st.session_state.memory


if ("chain" not in st.session_state
    or 
    st.session_state.TEMPLATE != TEMPLATE):
    st.session_state.chain = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
        } | 
    LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
)
chain = st.session_state.chain

sp_mapper = {"human":"student","ai":"patient", "user":"student","assistant":"patient"}

## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
## Grader part
index_name = f"indexes/{st.session_state.index_selectbox}/Rubric"

if "store2" not in st.session_state:
    st.session_state.store2 = db.get_store(index_name, embeddings=embeddings)
store2 = st.session_state.store2

if "TEMPLATE2" not in st.session_state:
    with open('templates/grader.txt', 'r') as file: 
        TEMPLATE2 = file.read()
    st.session_state.TEMPLATE2 = TEMPLATE2

with st.expander("Grader Prompt"):
    TEMPLATE2 = st.text_area("Grader Prompt", value=st.session_state.TEMPLATE2)

prompt2 = PromptTemplate(
    input_variables = ["question", "context", "history"],
    template = TEMPLATE2
)
if "retriever2" not in st.session_state:
    st.session_state.retriever2 = store2.as_retriever(search_type="similarity", search_kwargs={"k":2})
retriever2 = st.session_state.retriever2

def format_docs(docs):
    return "\n--------------------\n".join(doc.page_content for doc in docs)


# fake_history = '\n'.join([(sp_mapper.get(i.type, i.type) + ": "+ i.content) for i in memory.chat_memory.messages])
fake_history = '\n'.join([(sp_mapper.get(i['role'], i['role']) + ": "+ i['content']) for i in st.session_state.messages_1])
st.write(fake_history)

def y(_): 
    return fake_history

if ("chain2" not in st.session_state
    or 
    st.session_state.TEMPLATE2 != TEMPLATE2):
    st.session_state.chain2 = (
    {
        "context": retriever | format_docs, 
        "history": y,
        "question": RunnablePassthrough(),
        } | 

        # LLMChain(llm=llm_i, prompt=prompt2, verbose=False ) #|
        LLMChain(llm=llm_gpt4, prompt=prompt2, verbose=False ) #|
        | {
            "json": itemgetter("text"),
            "text": (
                LLMChain(
                    llm=llm, 
                    prompt=PromptTemplate(
                        input_variables=["text"],
                        template="Interpret the following JSON of the student's grades, and do a write-up for each section.\n\n```json\n{text}\n```"),
                        verbose=False)
                )
    }
)
chain2 = st.session_state.chain2

## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
## Streamlit now

# from dotenv import load_dotenv
# import os
# load_dotenv()
# key = os.environ.get("OPENAI_API_KEY")
# client = OpenAI(api_key=key)


if st.button("Clear History and Memory", type="primary"):
    st.session_state.messages_1 = []
    st.session_state.messages_2 = []
    st.session_state.memory = ConversationBufferWindowMemory(llm=llm, memory_key="chat_history", input_key="question" )
    memory = st.session_state.memory

## Testing HTML
# html_string = """
# <canvas></canvas>


# <script>
#     canvas = document.querySelector('canvas');
#     canvas.width = 1024;
#     canvas.height = 576;
#     console.log(canvas);

#     const c = canvas.getContext('2d');
#     c.fillStyle = "green";
#     c.fillRect(0,0,canvas.width,canvas.height);

#     const img = new Image();
#     img.src = "./tksfordumtrive.png";
#     c.drawImage(img,  10, 10);
# </script>

# <style>
#     body {
#         margin: 0;
#     }
# </style>
# """
# components.html(html_string,
#                 width=1280,
#                 height=640)


st.write("Timer has been removed, switch with this button")

if st.button(f"Switch to {'PATIENT' if st.session_state.active_chat==2 else 'GRADER'}"+".... Buggy button, please double click"):
    st.session_state.active_chat = 3 - st.session_state.active_chat

# st.write("Currently in " + ('PATIENT' if st.session_state.active_chat==2 else 'GRADER'))

# Create two columns for the two chat interfaces
col1, col2 = st.columns(2)

# First chat interface
with col1:
    st.subheader("Student LLM")
    for message in st.session_state.messages_1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Second chat interface
with col2:
    # st.write("pls dun spam this, its tons of tokens cos chat history")
    st.subheader("Grader LLM")
    st.write("grader takes a while to load... please be patient")
    for message in st.session_state.messages_2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Timer and Input
# time_left = None
# if st.session_state.start_time:
#     time_elapsed = datetime.datetime.now() - st.session_state.start_time
#     time_left = datetime.timedelta(minutes=10) - time_elapsed
#     st.write(f"Time left: {time_left}")

# if time_left is None or time_left > datetime.timedelta(0):
#     # Chat 1 is active
#     prompt = st.text_input("Enter your message for Chat 1:")
#     active_chat = 1
#     messages = st.session_state.messages_1
# elif time_left and time_left <= datetime.timedelta(0):
#     # Chat 2 is active
#     prompt = st.text_input("Enter your message for Chat 2:")
#     active_chat = 2
#     messages = st.session_state.messages_2

if st.session_state.active_chat==1:
    text_prompt = st.text_input("Enter your message for PATIENT")
    messages = st.session_state.messages_1
else:
    text_prompt = st.text_input("Enter your message for GRADER")
    messages = st.session_state.messages_2


from langchain.callbacks.manager import tracing_v2_enabled
from uuid import uuid4
import os

os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_ENDPOINT']='https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY']='ls__4ad767c45b844e6a8d790e12f556d3ca'
os.environ['LANGCHAIN_PROJECT']='streamlit'


if text_prompt:
    messages.append({"role": "user", "content": text_prompt})
    
    with (col1 if st.session_state.active_chat == 1 else col2):
        with st.chat_message("user"):
            st.markdown(text_prompt)
    
    with (col1 if st.session_state.active_chat == 1 else col2):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with tracing_v2_enabled(project_name = "streamlit"):
                if st.session_state.active_chat==1:
                    full_response = chain.invoke(text_prompt).get("text")
                else:
                    full_response = chain2.invoke(text_prompt).get("text").get("text")
            message_placeholder.markdown(full_response)
            messages.append({"role": "assistant", "content": full_response})


# import streamlit as st
# import time
# def count_down(ts):
#     with st.empty():
#         while ts:
#             mins, secs = divmod(ts, 60)
#             time_now = '{:02d}:{:02d}'.format(mins, secs)
#             st.header(f"{time_now}")
#             time.sleep(1)
#             ts -= 1
# st.write("Time Up!")
# def main():
#     st.title("Pomodoro")
#     time_minutes = st.number_input('Enter the time in minutes ', min_value=1, value=25)
#     time_in_seconds = time_minutes * 60
#     if st.button("START"):
#             count_down(int(time_in_seconds))
# if __name__ == '__main__':
#     main()
            
st.write('fake history is:')
st.write(y(""))
st.write('done')
