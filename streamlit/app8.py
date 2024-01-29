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
import db_firestore as db


## ----------------------------------------------------------------
## LLM Part
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory

import os, dotenv
from dotenv import load_dotenv
load_dotenv()


## Load from streamlit!!
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or st.secrets["HF_TOKEN"]
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["FIREBASE_CREDENTIAL"] = os.environ.get("FIREBASE_CREDENTIAL") or st.secrets["FIREBASE_CREDENTIAL"]


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages_1" not in st.session_state:
    st.session_state.messages_1 = []

if "messages_2" not in st.session_state:
    st.session_state.messages_2 = []

if "start_time" not in st.session_state:
    st.session_state.start_time = None

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
    st.session_state.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = st.session_state.llm
if "llm_gpt4" not in st.session_state:
    st.session_state.llm_gpt4 = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
llm_gpt4 = st.session_state.llm_gpt4

## ------------------------------------------------------------------------------------------------
## Patient part

index_name = "indexes/Headache/QA"

if "store" not in st.session_state:
    st.session_state.store = db.get_store(index_name, embeddings=embeddings)
store = st.session_state.store

TEMPLATE = """You are a patient undergoing a medical check-up. You will be given the following:
1. A context to answer the doctor, for your possible symptoms.
2. A question about your current symptoms.

Your task is to answer the doctor's questions as simple as possible, acting like a patient.
Do not include other symptoms that are not included in the context, which provides your symptoms.

Answer the question to the point, without any elaboration if you're not prodded with it.

As you are a patient, you do not know any medical jargon or lingo. Do not include specific medical terms in your reply.
You only know colloquial words for medical terms. 
For example, you should not reply with "dysarthria", but instead with "cannot speak properly". 
For example, you should not reply with "syncope", but instead with "fainting". 

Here is the context:
{context}

----------------------------------------------------------------
You are to reply the doctor's following question, with reference to the above context.
Question:
{question}
----------------------------------------------------------------
Remember, answer in a short and sweet manner, don't talk too much.
Your reply:
"""
if "TEMPLATE" not in st.session_state:
    st.session_state.TEMPLATE = TEMPLATE

with st.expander("Patient Prompt"):
    TEMPLATE = st.text_area("Patient Prompt", value=TEMPLATE)

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
    st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", input_key="question" )
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

sp_mapper = {"human":"student","ai":"patient"}

## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
## Grader part
index_name = "indexes/Headache/Rubric"

# store = FAISS.load_local(index_name, embeddings)

if "store2" not in st.session_state:
    st.session_state.store2 = db.get_store(index_name, embeddings=embeddings)
store2 = st.session_state.store2

TEMPLATE2 = """You are a teacher for medical students. You are grading a medical student on their OSCE, the Object Structured Clinical Examination.

Your task is to provide an overall assessment of a student's diagnosis, based on the rubrics provided.
You will be provided with the following information:
1. The rubrics that the student should be judged based upon.
2. The conversation history between the medical student and the patient.
3. The final diagnosis that the student will make.

=================================================================

Your task is as follows:
1. Your grading should touch on every part of the rubrics, and grade the student holistically.
Finally, provide an overall grade for the student.

Some additional information that is useful to understand the rubrics:
- The rubrics are segmented, with each area separated by dashes, such as "----------" 
- There will be multiple segments on History Taking. For each segment, the rubrics and corresponding grades will be provided below the required history taking.
- For History Taking, you are to grade the student based on the rubrics, by checking the chat history between the patients and the medical student.
- There is an additional segment on Presentation, differentials, and diagnosis. The 


=================================================================


Here are the rubrics for grading the student:
<rubrics>

{context}

</rubrics>

=================================================================
You are to give a comprehensive judgement based on the student's diagnosis, with reference to the above rubrics.

Here is the chat history between the medical student and the patient:

<history>

{history}

</history>
=================================================================


Student's final diagnosis:
<diagnosis>
    {question}
</diagnosis>

=================================================================

Your grade:
"""
if "TEMPLATE2" not in st.session_state:
    st.session_state.TEMPLATE2 = TEMPLATE2

with st.expander("Grader Prompt"):
    TEMPLATE2 = st.text_area("Grader Prompt", value=TEMPLATE2)

prompt2 = PromptTemplate(
    input_variables = ["question", "context", "history"],
    template = TEMPLATE2
)
if "retriever2" not in st.session_state:
    st.session_state.retriever2 = store2.as_retriever(search_type="similarity", search_kwargs={"k":2})
retriever2 = st.session_state.retriever2

def format_docs(docs):
    return "\n--------------------\n".join(doc.page_content for doc in docs)


fake_history = '\n'.join([(sp_mapper.get(i.type, i.type) + ": "+ i.content) for i in memory.chat_memory.messages])

if "memory2" not in st.session_state:
    st.session_state.memory2 = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", input_key="question" )
memory2 = st.session_state.memory2

def x(_): 
    return fake_history

if ("chain2" not in st.session_state
    or 
    st.session_state.TEMPLATE2 != TEMPLATE2):
    st.session_state.chain2 = (
    {
        "context": retriever | format_docs, 
        "history": x,
        "question": RunnablePassthrough(),
        } | 

    LLMChain(llm=llm, prompt=prompt2, memory=memory, verbose=False)
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

st.title("UAT for PatientLLM and GraderLLM")
st.title("Headache only, for now")

if st.button("Clear History and Memory", type="primary"):
    st.session_state.messages_1 = []
    st.session_state.messages_2 = []
    st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", input_key="question" )
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


if text_prompt:
    messages.append({"role": "user", "content": text_prompt})
    
    with (col1 if st.session_state.active_chat == 1 else col2):
        with st.chat_message("user"):
            st.markdown(text_prompt)
    
    with (col1 if st.session_state.active_chat == 1 else col2):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if st.session_state.active_chat==1:
                full_response = chain.invoke(text_prompt).get("text")
            else:
                full_response = chain2.invoke(text_prompt).get("text")
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
            
